# torch 是 PyTorch：Tensor、布尔矩阵、设备与张量运算；本文件不是原始数据读取器。
import torch
# nn=neural network，ModuleList 注册层集合，使参数保存、转设备与优化器收集可追踪。
from torch import nn

from uniperceiver.config import configurable
# 两个点表示上一级 modeling 包；这里导入项目自己的稠密编码层，内部使用 PyTorch MultiheadAttention。
from ..layers.transformer_encoder_layer import TransformerEncoderLayer
# MoE=Mixture of Experts（专家混合）；按配置选择稠密层或专家层，不是同时运行两套完整骨干。
from ..layers.transformer_encoder_moe_layer import MoETransformerEncoderLayer
from .build import ENCODER_REGISTRY
import uniperceiver.utils.comm as comm



__all__ = ["UnifiedBertEncoder"]

# construct=构建，attention=注意力；返回位置之间的可见性矩阵，或 None。
# data 是已经嵌入的特征，不是字符串；sample_info 记录各段长度，task_info 指明当前任务。
def _construct_attention_masks( data, sample_info, task_info):
    # bool 只存 True/False；此处沿用 nn.MultiheadAttention 约定：True=禁止关注，False=允许关注。
    # 这与某些其他库 1=允许 的 attention_mask 语义不同，不能机械复用。
    mask_type = torch.bool
    # device 指 CPU/GPU 设备；掩码与数据放同一设备，避免跨设备错误。
    device = data.device

    # None 表示本函数不施加位置对位置的结构约束；另外的 invalid_mask 仍可能屏蔽 PAD。
    attn_mask = None
    if isinstance(sample_info, list):
        sample_info = sample_info[0]
    # caption=描述生成；text_spe_cat 标识“文字段 + 预测 SPE 段”拼接格式。
    # get(key,False) 缺键时返回 False；仅满足任务与格式两个条件才创建这里的特殊掩码。
    if task_info['task_type'] in ['image_caption', 'video_caption'] and sample_info.get('text_spe_cat', False):

        # the extra 1 length for spe token
        # 拆包得到三个整数：首部 SPE 长度、视觉 token 长度、文字与预测段总长度。
        # img=image 的缩写，此处视觉段也可以来自视频；它不是图片像素边长。
        spe_length, img_length, text_total_length = sample_info['data_length']
        # // 是整除；约定两个文字相关段等长，得到每段 T。不是随意把任何文字长度减半。
        text_length = text_total_length//2

        # ones 创建全 True 的 [L,L] 矩阵，即先禁止全部位置互相关注，再逐块开放。
        # L=spe_length+img_length+2*T；行是 query 位置，列是 key 位置。
        attn_mask = torch.ones((spe_length + img_length + text_total_length,
                                spe_length + img_length + text_total_length), dtype=mask_type, device=device)

        # 所有行都可读首部 SPE/视觉段：这块列填 False。
        # 后续文字列仍默认禁止，因而视觉段不会经当前掩码读到真实未来文字。
        attn_mask[:spe_length + img_length + text_total_length, :spe_length+img_length] = False
        # 真实文字段内部使用 triu(diagonal=1)：主对角线之上的未来位置禁止。
        # 当前与过去文字允许；真实词本身可见，但这个段不是直接拿来预测它自己的标签。
        attn_mask[spe_length + img_length:spe_length + img_length + text_length, spe_length + img_length:spe_length + img_length + text_length] =  torch.ones(
                (text_length, text_length),  dtype=mask_type, device=device).triu_(diagonal=1)
        # 预测 SPE 段读取真实文字时用 triu(diagonal=0)，连同位置真实词也禁止。
        # 第 i 个预测 SPE 只能看真实文字 0..i-1，避免把正确答案直接泄漏给模型。
        attn_mask[spe_length + img_length + text_length:, spe_length + img_length:spe_length + img_length + text_length] =  torch.ones(
                (text_length, text_length),
                dtype=mask_type,
                device=device).triu_(diagonal=0)
        # 下面设置预测段内部：只允许各 SPE 读自己的位置，禁止读其他预测 SPE。
        # diag 构造单位对角矩阵，~ 是布尔取反，得对角 False、其他 True。
        attn_mask[spe_length + img_length + text_length:,
                  spe_length + img_length + text_length:] = ~torch.ones(
                      (text_length), dtype=mask_type,
                      device=device).diag()

    return attn_mask


@ENCODER_REGISTRY.register()
# Unified=统一，BERT=Bidirectional Encoder Representations from Transformers，Encoder=编码器。
# 任务掩码可以使共享骨干具备因果可见性，类名 BERT 不意味着所有任务都双向读未来。
class UnifiedBertEncoder(nn.Module):
    @configurable
    def __init__(self, *, num_hidden_layers: int, bert_layers,
                 skip_target_encode, word_balance_losses,
                 bookswiki_word_alone, cfg):
        super(UnifiedBertEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        self.skip_target_encode = skip_target_encode
        self.word_balance_losses = word_balance_losses
        self.bookswiki_word_alone = bookswiki_word_alone
        self.cfg = cfg



    @classmethod
    def from_config(cls, cfg):
        # DropPath=随机丢弃残差分支；FIXED 表示所有层用同一概率。
        if cfg.MODEL.BERT.DROP_PATH_PROB_FIXED:
            dpr = [cfg.MODEL.BERT.DROP_PATH_PROB for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        else:
            # dpr=drop path rates，各层丢弃率；linspace 生成线性间隔数，item 转 Python 标量。
            # 非固定时从 0 逐层增加到配置概率；不是为每个 token 设置不同概率。
            dpr = [x.item() for x in torch.linspace(0, cfg.MODEL.BERT.DROP_PATH_PROB, cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]

        layers = []
        # idx=index（编号），从 0 开始；NUM_HIDDEN_LAYERS 是层数，不是隐藏维度。
        for layer_idx in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS):
            # MOE 开关关闭时走原版稠密 Transformer；下面配置 d_model=D，nhead=头数 H。
            # 通常每头维度 D/H 要整除；FFN 中间维度另由 INTERMEDIATE_SIZE 决定。
            if not cfg.MOE.MOE:
                layers.append(
                    TransformerEncoderLayer(
                        d_model=cfg.MODEL.BERT.HIDDEN_SIZE,
                        nhead=cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
                        dim_feedforward=cfg.MODEL.BERT.INTERMEDIATE_SIZE,
                        dropout=cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB,
                        drop_path_ratio=dpr[layer_idx],
                        activation=cfg.MODEL.BERT.HIDDEN_ACT,
                        layer_scale=cfg.MODEL.LAYER_SCALE,
                        ls_init_values=cfg.MODEL.LAYER_SCALE_INIT,
                        batch_first=True,
                        norm_first=True,
                        cfg = cfg,
                    ))
            else:
                attention_moe = False
                ffn_moe = False

                moe_layer_start_idx = cfg.MOE.MOE_LAYER_START_IDX
                moe_layer_end_idx = cfg.MOE.MOE_LAYER_END_IDX

                if cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'odd':
                    if layer_idx % 2 == 0 and layer_idx >= moe_layer_start_idx and layer_idx < moe_layer_end_idx:
                        moe_layers = cfg.MOE.MOE_EXPERT_TYPE.split(',')
                        attention_moe = "SA" in moe_layers
                        ffn_moe = 'FFN' in moe_layers

                elif cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'four':
                    if layer_idx % 4 == 0 and layer_idx >= moe_layer_start_idx and layer_idx < moe_layer_end_idx:
                        moe_layers = cfg.MOE.MOE_EXPERT_TYPE.split(',')
                        attention_moe = "SA" in moe_layers
                        ffn_moe = 'FFN' in moe_layers

                elif cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'all':
                    if layer_idx >= moe_layer_start_idx and layer_idx < moe_layer_end_idx:
                        moe_layers = cfg.MOE.MOE_EXPERT_TYPE.split(',')
                        attention_moe = "SA" in moe_layers
                        ffn_moe = 'FFN' in moe_layers
                elif cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'none':
                    attention_moe = None
                    ffn_moe = None


                elif cfg.MOE.MOE:
                    raise NotImplementedError('cfg.MOE.MOE_EXPERT_LOCATION')

                layers.append(
                    MoETransformerEncoderLayer(
                        d_model=cfg.MODEL.BERT.HIDDEN_SIZE,
                        nhead=cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
                        dim_feedforward=cfg.MODEL.BERT.INTERMEDIATE_SIZE,
                        dropout=cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB,
                        drop_path_ratio=dpr[layer_idx],
                        activation=cfg.MODEL.BERT.HIDDEN_ACT,
                        layer_scale=cfg.MODEL.LAYER_SCALE,
                        ls_init_values=cfg.MODEL.LAYER_SCALE_INIT,
                        batch_first=False,
                        norm_first=True,
                        cfg = cfg,
                        ffn_moe=ffn_moe,
                        attn_moe=attention_moe,
                    ))



        # ModuleList 与普通 list 不同：其中各层被登记为子模块，训练/存盘可找到其参数。
        bert_layers = nn.ModuleList(
            layers
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "skip_target_encode": cfg.MODEL.BERT.SKIP_TARGET_ENCODE,
            "bert_layers": bert_layers,
            "word_balance_losses": cfg.SOLVER.WORD_BALANCE_LOSSESS,
            "bookswiki_word_alone": cfg.MODEL.BW_WORD_ALONE,
            "cfg": cfg
        }

    @classmethod
    def add_config(cls, cfg):
        pass


    # forward 是实例前向方法，由 model(...) 间接调用；data 通常 [B,L,D]。
    # invalid_mask 是每条样本的无效 key 位 [B,L]；history_states 是逐层历史特征，不是原文本。
    # return_all=True 返回嵌入及每层输出列表；False 只返回最后一层 Tensor。
    def forward(self, data, invalid_mask, sample_info, task_info, history_states=None, return_all=False, **kwargs):

        attn_mask = _construct_attention_masks(data, sample_info, task_info)
        kwargs.update({'sample_info': sample_info})
        data_type = kwargs.get('data_type', 'input')
        # 此开关跳过候选目标的编码，仅返回输入嵌入；原注释用于调试，不能说与正常共享编码完全等价。
        if data_type == 'target' and self.skip_target_encode:
            # used for debugging with single gpu sometimes
            return data 
        if return_all:
            data_all = [data]
        # enumerate 同时得到层编号 l 和层对象；每层输出覆盖 data，形状通常仍为 [B,L,D]。
        for l, layer_module in enumerate(self.layers):

            if history_states is None:
                data = layer_module(src=data, src_mask=attn_mask, src_key_padding_mask=invalid_mask, task_info=task_info, **kwargs)
            else:
                data = layer_module(src=data, src_mask=attn_mask, src_key_padding_mask=invalid_mask, history_states=history_states[l], task_info=task_info, **kwargs)

            if return_all:
                data_all.append(data)

        # 条件表达式：False 返回 Tensor，True 返回 list；调用者必须了解类型会随开关变化。
        return data if not return_all else data_all
