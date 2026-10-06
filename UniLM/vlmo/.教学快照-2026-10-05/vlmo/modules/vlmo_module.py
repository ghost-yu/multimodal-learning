# os 用于文件路径；后续读取和保存模型检查点会用到。
import os
# torch 负责张量计算和 checkpoint 加载。
import torch
# nn 提供 Embedding、Linear、Sequential 等可训练模块。
import torch.nn as nn
# F.embedding 依据位置索引查相对位置偏置表。
import torch.nn.functional as F
# LightningModule 提供训练、验证和优化器的标准接口。
import pytorch_lightning as pl
# NumPy 参与温度参数初始化与位置插值坐标计算。
import numpy as np
# 导入时会注册 vlmo_base_patch16 等模型名，供 timm.create_model 查找。
import vlmo.modules.multiway_transformer

# BERT 配置与词嵌入层负责把 token ID 变成文字向量。
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
# heads 是各任务的小分类头；objectives 是损失；vlmo_utils 处理训练日志与调度。
from vlmo.modules import heads, objectives, vlmo_utils
# 只让主进程打印，避免多卡重复输出。
from pytorch_lightning.utilities.distributed import rank_zero_info
# 改变图像分辨率时，对旧的位置偏置做插值。
from scipy import interpolate
# 按注册的模型名创建视觉 Transformer。
from timm.models import create_model


def convert_to_textpt_ckpt(state_dict, module):
    """把已有视觉权重键名整理为 VLMo 文字预训练阶段可加载的格式。"""
    # 新字典将保存转换后的参数名与张量。
    new_state_dict = {}

    # Merge relative_position_bias_table from all layer into one tensor, 
    # so we can use one op for gather the relative position bias for speed up
    # 先收集各层单独存储的相对位置偏置。
    relative_position_bias_tables = {}

    # 逐项检查旧 checkpoint 的每个参数名。
    for key in state_dict:
        # value 就是当前参数张量。
        value = state_dict[key]

        if "relative_position_bias_table" in key:
            # transformer.blocks.0.attn.relative_position_bias_table
            # 从 key 里解析 Transformer 层号，例如 blocks.0。
            layer_idx = int(key.split(".attn.")[0].split('.')[-1])
            # 先按层号暂存，稍后按正确顺序拼接。
            relative_position_bias_tables[layer_idx] = value
            # 这一项会作为合并后的新键保存，不直接复制原键。
            continue

        if "mlp" in key:
            # 原视觉 MLP 映射到 VLMo 的 image 专家分支。
            key_imag = "transformer." + key.replace("mlp", "mlp_imag")
            new_state_dict[key_imag] = value
        elif "norm2" in key:
            # 与视觉 FFN 配套的第二个 LayerNorm 也映射到 image 分支。
            key_imag = "transformer." + key.replace("norm2", "norm2_imag")
            new_state_dict[key_imag] = value
        else:
            # 共享注意力等其余参数只需添加 transformer. 前缀。
            new_key = "transformer." + key
            new_state_dict[new_key] = value
    
    if len(relative_position_bias_tables) > 0:
        # 按层号排列，确保第 0 层偏置不会被接到错误位置。
        tensor_list = []
        for layer_idx in sorted(relative_position_bias_tables.keys()):
            tensor_list.append(relative_position_bias_tables[layer_idx])
        # dim=1 把各层注意力头的偏置沿列方向合并。
        relative_position_bias_table = torch.cat(tensor_list, dim=1)

        # 旧表实际拥有的相对距离种类数。
        num_distence, _ = relative_position_bias_table.shape
        # 克隆新模型当前表，保留旧 checkpoint 未覆盖的额外位置。
        all_relative_position_bias_table = module.relative_position_bias_table.data.clone()
        # 将旧视觉偏置填入新表的前面几行。
        all_relative_position_bias_table[:num_distence, :] = relative_position_bias_table

        # 用 VLMo 新参数名保存合并后的偏置表。
        new_state_dict["relative_position_bias_table"] = all_relative_position_bias_table
        
    return new_state_dict


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    """图片分辨率改变时，保持 CLS 不动、重采样 patch 位置向量。"""
    # interpolate position embedding
    # 最后一维是每个位置向量的隐藏维度 D。
    embedding_size = pos_embed_checkpoint.shape[-1]
    # 新视觉编码器有多少个图像 patch。
    num_patches = visual_encoder.patch_embed.num_patches
    # 位置向量总数减去 patch 数，就是 CLS 等额外 token 数。
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        # CLS 不属于二维 patch 网格，不参与图像插值。
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        # 剩余位置向量按旧网格尺寸重排。
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # 插值函数接收 [B,C,H,W]，所以把特征维移到第二轴。
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        # 双三次插值生成目标网格高和宽。
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # 再换回 [B,patch 数,D] 的 token 序列。
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # 把未改动的 CLS 位置向量接回序列开头。
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        # 网格大小相同时，直接使用旧 checkpoint 的位置向量。
        return pos_embed_checkpoint


def convert_deepspeed_ckpt(state_dict):
    """去掉 DeepSpeed 保存时可能添加的 module. 参数名前缀。"""
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("module."):
            # 例如 module.transformer.blocks.0 → transformer.blocks.0。
            new_key = key[len("module."):]
            value = state_dict[key]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


class VLMo(pl.LightningModule):
    """把数据、共享骨干、任务头和各项损失接成可训练模型。"""

    def __init__(self, config):
        # 初始化 LightningModule，并保存配置供别的方法读取。
        super().__init__()
        self.save_hyperparameters()

        # backbone & patch projection
        # 模型输入图片要缩放到这个边长。
        self.img_size = config["image_size"]
        # 根据 vlmo_base_patch16 等注册名建立多路 Transformer。
        self.transformer = create_model(
            config["model_arch"],
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            config=self.hparams.config,
        )
        # 后面的位置偏置计算需要 patch 大小、层数和特征宽度。
        self.patch_size = self.transformer.patch_size
        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        # 构建图像、文字和跨模态 token 的相对位置索引。
        self.build_relative_position_embed(config)
        
        # language embedding
        # BERT 的词表和隐藏维度必须与共享 Transformer 相容。
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.num_features,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"],
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute", 
        )

        # token ID、词位置等在这里变成 [B,T,D] 文字向量。
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        # 0 代表文字，1 代表图片；NLVR2 后续扩成 3 种类型。
        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(objectives.init_weights)

        # task layers        
        # Pooler 读取图文序列第 0 个位置，形成整体配对表示。
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)

        ## language modeling
        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
            # MLM 头把隐藏向量投影到词表大小，预测被遮盖的单词。
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        ## image-text matching (global hard negative)
        if config["loss_names"]["itm"] > 0:
            # ITM 头输出“图文不匹配/匹配”两个类别分数。
            self.itm_score = heads.ITMHead(self.num_features)
            self.itm_score.apply(objectives.init_weights)
        
        ## contrastive loss (or sampling for global hard negative)
        if config["loss_names"]["itc"] > 0:
            # 第一对投影头从单模态文字/图像 CLS 得到可比较的向量。
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            # 第二对投影头使用经过 VL-FFN 分支的文字/图像表示。
            self.itc_vl_text_proj = heads.ITCHead(self.num_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            # exp(logit_scale) 在相似度矩阵进入交叉熵前调整数值尺度。
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ## retrieval task ft
        if config["loss_names"]["irtr"] > 0:
            # 检索微调只需图像和文字两个单模态投影头。
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 若配置了 load_path，在训练前加载视觉/语言/图文阶段的已有权重。
        self.load_pretrained_weight()
        
        # ===================== Downstream ===================== #
        ## VQAv2
        if self.hparams.config["loss_names"]["vqa"] > 0:
            # VQA 从整体图文向量预测词表中的答案编号。
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        ## NLVR2 (Visual reasoning)
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            # NLVR2 一句话对应两张图片，因此拼接两份整体表示。
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.num_features * 2, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            # 保存原有文字/第一张图类型向量，扩容到第二张图时再复制。
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, self.num_features)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        # 创建训练/验证准确率、损失等统计对象。
        vlmo_utils.set_metrics(self)
        # 当前 batch 执行哪些目标，由 training_step 中的 set_task 设置。
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            # 纯测试模式需要加载下游任务已训练好的完整参数。
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")

            state_dict = None
            
            # 不同训练框架把参数字典放在不同键名下。
            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt

            # strict=False 允许不同任务头导致的缺失或额外参数，并打印名单。
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

    def load_pretrained_weight(self):
        """训练前加载阶段性权重，并处理参数名与位置向量的尺寸差异。"""
        # 仅在给了路径且不是纯测试模式时加载预训练权重。
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            config = self.hparams.config
            # 先把大权重载入 CPU，避免加载瞬间额外占满显存。
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))

            state_dict = None

            # Lightning、DeepSpeed 和普通训练脚本常用不同外层键名。
            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            # DeepSpeed 保存的 module. 前缀需要先移除。
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt

            # 打印各参数形状，检查 checkpoint 与当前结构是否一致。
            for key in state_dict:
                var = state_dict[key]
                rank_zero_info("%s = %s" % (key, str(var.size())))

            rank_zero_info(config["loss_names"])
            if config["loss_names"]["textmlm"] > 0:
                # 文字预训练阶段把纯视觉权重映射到 VLMo 图像专家。
                rank_zero_info("convert to textpt")
                state_dict = convert_to_textpt_ckpt(state_dict, self)

            # 文字最大长度可能由视觉预训练时的 196 改为图文训练时的 40。
            max_text_len = config["max_text_len"]
            if "text_embeddings.position_embeddings.weight" in state_dict and state_dict["text_embeddings.position_embeddings.weight"].size(0) != max_text_len:
                # 截去超出的文字位置向量及配套位置编号。
                state_dict["text_embeddings.position_embeddings.weight"].data = state_dict["text_embeddings.position_embeddings.weight"].data[:max_text_len, :]
                state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:, :max_text_len]
                rank_zero_info("text position_embeddings size: {}".format(state_dict["text_embeddings.position_embeddings.weight"].size()))
                # 旧长度对应的位置索引要丢弃，后续按新长度重新构造。
                for check_key in ("relative_position_index", "text_relative_position_index", "text_imag_relative_position_index"):
                    if check_key in state_dict:
                        state_dict.pop(check_key)

            if "transformer.pos_embed" in state_dict:
                # 图片分辨率变了，patch 网格也变；插值位置向量以适配新网格。
                pos_embed_reshaped = interpolate_pos_embed(state_dict['transformer.pos_embed'], self.transformer)         
                state_dict['transformer.pos_embed'] = pos_embed_reshaped

            if "relative_position_bias_table" in state_dict:
                # 相对位置偏置表也可能因图片分辨率而尺寸不同。
                rel_pos_bias = state_dict["relative_position_bias_table"]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.relative_position_bias_table.size()
                dst_patch_shape = self.transformer.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                # 除普通 patch-patch 距离外，还包括 CLS、跨模态等特殊位置。
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    # 与旧网格绑定的位置索引不能复用，先从权重字典移除。
                    state_dict.pop("relative_position_index")
                    state_dict.pop("text_relative_position_index")
                    state_dict.pop("text_imag_relative_position_index")
                    
                    rank_zero_info("Position interpolate from %dx%d to %dx%d" % (
                        src_size, src_size, dst_size, dst_size))
                    # 特殊位置不参与二维插值；单独暂存。
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    # 前半段 patch 相对位置偏置需要按新网格重采样。
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        # 几何级数控制插值采样点：中心密集，边缘逐渐稀疏。
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    # 二分搜索使旧坐标范围映射到目标网格范围。
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    # 从中心往正方向构造非均匀采样距离。
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    # 镜像得到负方向坐标，再把 0 接在中间。
                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    rank_zero_info("Original positions = %s" % str(x))
                    rank_zero_info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    # 每个注意力头都有独立二维偏置图，要逐头插值。
                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    # 按头维度合并插值后的各张二维偏置图。
                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    # 把没有插值的特殊位置偏置接回表尾。
                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    state_dict["relative_position_bias_table"] = new_rel_pos_bias

            # 非严格加载后打印缺失/额外键，确认任务头差异。
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

    def get_rel_pos_bias(self, relative_position_index):
        """把 token 两两位置索引变成每层、每头可用的偏置矩阵。"""
        if self.relative_position_embed:
            # F.embedding 把整数索引查成真实的、可学习的偏置数值。
            relative_position_bias = F.embedding(relative_position_index.long().to(self.relative_position_bias_table.device),
                                                    self.relative_position_bias_table)
            # 调整为 [所有层的注意力头,查询 token,被查询 token]。
            all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
            # 沿头维切成 num_layers 份，Block i 只拿到第 i 层的偏置。
            relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)
            return relative_position_bias_list
        else:
            # 不使用相对位置偏置时，每层都收到 None。
            return [None] * self.num_layers

    def build_relative_position_embed(self, config):
        """预先算出图像、文字与跨模态 token 的两两位置编号。"""
        if not self.transformer.need_relative_position_embed:
            # Base+ 等配置使用绝对位置向量，不建立相对位置索引。
            self.relative_position_embed = False
            self.text_imag_relative_position_index = None
            self.text_relative_position_index = None
            self.relative_position_index = None
            return
        # 图像 patch 是正方形网格，例如 224/16=14，窗口为 14×14。
        self.relative_position_embed = True
        window_size = (int(self.img_size / self.patch_size), int(self.img_size / self.patch_size)) #(14, 14)
        rank_zero_info("window_size: {}".format(window_size))
        # 每个注意力头都需要自己的偏置数值。
        num_heads = self.transformer.num_heads
        max_text_len_of_initckpt = config["max_text_len_of_initckpt"] #196
        max_text_len = config["max_text_len"] #40
        # +1 是图片 CLS token；14×14 时图像 token 共 197 个。
        max_imag_len = window_size[0] * window_size[1] + 1 #197
        self.window_size = window_size
        # 图像相对坐标的每个方向从 -(W-1) 到 +(W-1)，再加 3 个 CLS 专用位置。
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 文本偏置为不同的前后词距离预留编号。
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        # 最后 +2 是文字指向图像、图像指向文字两种跨模态关系。
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        # 每个距离编号在每层每个头都有一项可学习偏置。
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers))
        
        # get pair-wise relative position index for each token inside the window
        # 给 14×14 patch 网格构造行号和列号。
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # 第一张网格保存行坐标，第二张保存列坐标。
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 广播减法得到“任意两个 patch 之间的行/列差”。
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 加 W-1，把负距离整体平移到非负整数索引。
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        # 行差乘以列差类别总数，再加列差，相当于把二维坐标编码成一维编号。
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        # 第 0 行/列留给图像 CLS；其余填普通 patch-patch 距离。
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # CLS→patch、patch→CLS、CLS→CLS 分别使用不同特殊编号。
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index
        
        # 文本第 0 个位置是 CLS，后续普通词位置另算相对距离。
        text_position_ids = torch.arange(max_text_len-1)
        # 广播减法算任意两个文字 token 的先后距离。
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        # rank_zero_info("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        # 文字 CLS 所在第 0 行/列同样单独预留。
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index
        
        # 跨模态没有共同的二维坐标，用两个专门编号表示方向。
        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        # 左上文本-文本、右上文本-图像、左下图像-文本、右下图像-图像。
        text_row_relative_position_index = torch.cat((text_relative_position_index, text2imag_relative_position_index), 1)
        imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, relative_position_index), 1)
        text_imag_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0)
        self.text_imag_relative_position_index = text_imag_relative_position_index

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        """图文一起输入：共享注意力融合，返回文本、图像和整体特征。"""
        # NLVR2 等任务可能把第二张图命名为 image_1。
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        # MLM 时取遮词版 text_ids_mlm；否则取原始 text_ids。
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        # 文字 mask 中 1 表示真实词，0 表示 PAD。
        text_masks = batch[f"text_masks"]
        # 文本 token ID [B,T] -> 文字向量序列 [B,T,D]。
        text_embeds = self.text_embeddings(text_ids)

        # 数据整理器把图片张量包在一个列表中，这里取第 0 个视图。
        img = batch[imgkey][0]
        # 图片切 patch 得到图像 token；mask 标出有效 patch。
        image_embeds, image_masks = self.transformer.visual_embed(img)

        # 把图像有效位移到同一张 GPU，再转成长整型供 Embedding 使用。
        image_masks = image_masks.long().to(device=img.get_device())
        # 文字加类型 0，图片加类型 image_token_type_idx (通常为 1)。
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        # 图文沿序列维拼接成 [B,T+P,D]，供共享注意力同时处理。
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # 注意力 mask 也按完全相同的 token 顺序拼接。
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        # 相对位置偏置向注意力提供 token 之间的位置关系信息。
        relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

        # 各层共享注意力结构相同，但参数按层独立。
        for i, blk in enumerate(self.transformer.blocks):
            # vl 让 Block 选择图文输入对应的 FFN 路径；注意力仍是共享的。
            x = blk(x, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])

        # 最后一层输出先归一化，再拆回图文两段。
        x = self.transformer.norm(x)
        # 前 T 个是文字，后 P 个是图片，保持原输入顺序。
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        # 第 0 个位置经 Pooler 得到整个图文对的向量，供 ITM/VQA 使用。
        cls_feats = self.pooler(x)

        # 字典让不同目标函数按名字取出自己所需的向量和标签。
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image": img,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        # 返回图文 token、整体特征和原始输入信息。
        return ret

    def infer_text(
        self,
        batch,
        mask_text=False,
    ):
        """预训练 ITC：文字独立编码，并额外计算经过 VL-FFN 的表示。"""
        # 有 mask_text 时读取被遮词版本；普通 ITC 使用原始文字。
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # BERT 词嵌入输出 [B,T,D]。
        text_embeds = self.text_embeddings(text_ids)
        # 额外加入模态类型 0，表示这些 token 属于文字。
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        # 保存各层结果，以便从第一个 VL-FFN 层前分出另一条路径。
        all_hidden_states = []
        # 文字-文字 token 的相对位置偏置，每层各取一份。
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            # 模态为 text，每层使用文字 FFN；注意力参数依旧是该层共享的。
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)
        
        # 从 VL-FFN 开始层的前一层输出分叉，再用 vl 模式重走后段。
        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])

        # 原始 text 专家路径的最后一层输出。
        lffn_hiddens = all_hidden_states[-1]

        # 两条路径各自取 CLS，经过不同投影头并做单位长度归一化。
        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        # 单模态 CLS → 文本 ITC 向量，用于第一套对比损失。
        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        # VL-FFN 分支 CLS → 第二套文本 ITC 向量。
        cls_vlffn_feats = self.itc_vl_text_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_ft(
        self,
        batch,
        mask_text=False,
    ):
        """检索微调版文字编码：只取 text 专家路径，不重走 VL-FFN。"""
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        # 保留最后一层文字特征供投影头读取。
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        # 取第 0 个 CLS token，映射成单位长度文本向量。
        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_mlm(
        self,
        batch,
        mask_text=False,
    ):
        """文字单模态 MLM：返回每个词的特征，供 mlm_score 预测词 ID。"""
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # 被遮词的 token ID 先转成 [B,T,D] 向量。
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        # 这里只需要文字 token 之间的相对位置偏置。
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        # 返回各个词的最终特征；没有输入图片，因此 image_feats=None。
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": None,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_image(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        """预训练 ITC：图片独立编码，同时得到 image 与 VL-FFN 两套向量。"""
        # 某些任务使用 image_0/image_1；一般图文任务使用 image。
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        # 列表第 0 个元素是数据整理器产生的图片视图 [B,3,H,W]。
        img = batch[imgkey][0]
        # 图像 patch 加 CLS，得到 [B,P+1,D]。
        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        # 类型向量区分图片与文字；NLVR2 的第二张图可用类型 2。
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        # 每层的 image 专家输出保留起来，供分叉到 VL-FFN。
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        # 所有层先走纯 image 路径。
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)
        
        # 从 VL-FFN 启用前的位置分叉，再在最后几层重走 vl 路径。
        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])
        
        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        # image 专家路径的 CLS → 第一套图像对比向量。
        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        # VL-FFN 分支的 CLS → 第二套图像对比向量。
        cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def infer_image_ft(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        """检索微调版图片编码：只取 image 专家路径的最终向量。"""
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        img = batch[imgkey][0]
        # 从图片像素得到 patch＋CLS 序列及其有效位置表。
        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        # 第 0 个图像 CLS 投影并归一化，后续直接与文字向量点积。
        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def forward(self, batch):
        """按 current_tasks 路由损失；同一 batch 可同时做 MLM、ITC、ITM。"""
        # ret 汇总各任务函数返回的损失、预测和日志字段。
        ret = dict()
        # 如果没有启用任务，只返回普通图文融合结果。
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Textonly Masked Language Modeling
        if "textmlm" in self.current_tasks:
            ret.update(objectives.compute_textonly_mlm(self, batch))

        # Contrastive loss for pretraining
        # ITC 要先于 ITM，后者会读取前者返回的相似度来抽难负例。
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # Contrastive loss for finetuning
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        # Image Text Matching with global hard negative, must use with itc
        # 错配样本概率由 ret 中的图找文/文找图相似度决定。
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_hardneg(self, batch, ret["itc_i2t_logits"], ret["itc_t2i_logits"]))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        # 从配置的 loss_names 得到本次要执行的任务列表。
        vlmo_utils.set_task(self)
        # 调用 forward 计算任务相关输出。
        output = self(batch)
        # 只把字段名含 loss 的数值相加；Lightning 会对此总损失反传。
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        # 汇总一轮训练的损失和准确率。
        vlmo_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        # 验证时执行同一套目标，但不做参数更新。
        vlmo_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        # 测试集需要时可额外收集 VQA 答案。
        vlmo_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.hparams.config["log_dir"])
        vlmo_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        # 学习率 warmup、衰减与参数组由工具函数统一构造。
        return vlmo_utils.set_schedule(self)
