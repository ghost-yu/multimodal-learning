# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

# torch=PyTorch 主库：cat/split 操作张量，ones 创建标量初值，自动求导记录各层参数的关系。
import torch
# nn 是神经网络模块库：Module/Linear/LayerNorm/GELU/Sequential/Identity 在本文件构造各任务头。
# Linear 把最后一维投影为新宽度；Sequential 顺序执行各层；Identity 原样返回，不引入参数。
import torch.nn as nn
# functional 提供函数式操作，惯例缩写 F；normalize 把向量 L2 长度归一化为 1，用于余弦检索。
import torch.nn.functional as F
# register_model 是 timm 的注册装饰器，登记下面的工厂函数，使入口 create_model 可按名字找到它。
from timm.models.registry import register_model
# NumPy 数值库，np 是约定简称；本文件用 np.log 求自然对数，不用 NumPy 反向训练网络。
import numpy as np

# utils 是同目录项目工具模块，不是 Python 标准库。含分布式 rank/world_size 与 ClipLoss。
import utils
from modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config


class TwoLayerMLP(nn.Module):
    # TwoLayer=两层，这里指两次 Linear；MLP=multi-layer perceptron（多层感知机）。
    # in/hidden/out_features 分别是输入、中间、输出的最后一维宽度，前面的 batch 维保持不变。
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        # 条件表达式：norm_input 为 True 就归一化输入，否则用 Identity 保持接口一致。
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    # Pooler=池化器。这里实际取第 0 个 CLS 再投影，不是 CoCa 的可学习注意力 query 池化。
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        # cls_rep=CLS representation（汇总 token 表示）；x=[B,L,D]，取第 0 位置得到 [B,D]。
        # 切片第一个冒号保留全部样本，0 取一个位置，最后冒号保留全部特征。
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        # Tanh 将数值压到 (-1,1)，是一种非线性激活；不表示类别概率，也不是 softmax。
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForVisualReasoning(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualReasoning, self).__init__(args=args)
        # embed_dim=embedding dimension（嵌入宽度）D；这里从骨干配置读，Base 为 768，Large 为 1024。
        embed_dim = args.encoder_embed_dim
        # 两张图各含“视觉 CLS + 文字 CLS”，最终拼成 4D 向量做二分类。
        self.head = TwoLayerMLP(
            in_features=embed_dim * 4, 
            hidden_features=embed_dim * 2,
            out_features=num_classes, 
            norm_layer=norm_layer, 
        )
        init_scale = 0.001
        self.head.apply(self._init_weights)
        if isinstance(self.head.dense1, nn.Linear):
            self.head.dense1.weight.data.mul_(init_scale)
            self.head.dense1.bias.data.mul_(init_scale)

        if isinstance(self.head.dense2, nn.Linear):
            self.head.dense2.weight.data.mul_(init_scale)
            self.head.dense2.bias.data.mul_(init_scale)

    def forward(self, image_a, image_b, text_description, padding_mask, **kwargs):
        # bsz=batch size，文字编号 [B,T] 的第 0 维；_ 是暂时不用的第二维长度，不是固定关键字。
        # text_description 仍为整数编号 Tensor，并不是一条 Python 字符串。
        bsz, _ = text_description.size()
        
        # 把两张图合成 2B 条“图+同一句话”输入，骨干只需调用一次。
        vision_input = torch.cat((image_a, image_b), dim=0)
        # 文字和 padding mask 也复制一份，与 2B 张图逐一对应。
        language_input = torch.cat((text_description, text_description), dim=0)
        padding_mask = torch.cat((padding_mask, padding_mask), dim=0)

        outputs = self.beit3(
            textual_tokens=language_input, 
            visual_tokens=vision_input, 
            text_padding_position=padding_mask, 
        )
        # outputs 是骨干返回的字典；encoder_out Tensor 的形状 [2B,I+T,D]，I 包括视觉 CLS。
        x = outputs["encoder_out"]
        # multiway_split_position=视觉/文字序列边界整数 I；文字第一个 token 位于拼接序列的第 I 项。
        multiway_split_position = outputs["multiway_split_position"]

        # 从骨干序列中分别取视觉 CLS 与文字 CLS。
        vision_cls = x[:, 0, :]
        language_cls = x[:, multiway_split_position, :]
        # cat 在最后一维拼接两种汇总表示：[2B,D]+[2B,D]→[2B,2D]，不是增加 batch 数。
        cls_rep = torch.cat((vision_cls, language_cls), dim=-1)
        # 前 B 条对应 image_a，后 B 条对应 image_b；切回两组。
        # split 给定两个段长 [B,B]，在样本维 dim=0 切开；a/b 对应图 a/图 b 与同一句话的融合表示。
        a, b = torch.split(cls_rep, split_size_or_sections=[bsz, bsz], dim=0)
        # 同一样本的两张图再次拼成 [B,4D]，交给任务头判断真假。
        cls_rep = torch.cat((a, b), dim=-1)
        return self.head(cls_rep)
    

class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def forward(self, image, **kwargs):
        # 纯视觉分类：textual_tokens=None，骨干全部走视觉分支；image 为 [B,3,H,W]。
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        # 去掉开头视觉 CLS，剩余 [B,P,D] 图片小块；此实现选择平均 patch，而非直接拿 CLS。
        t = x[:, 1:, :]
        # mean(1) 沿 token 轴平均→[B,D]，fc_norm 做特征归一化；后面 head 输出 [B,类别数]。
        # 这与当前论文 Giant 分类描述的“图片与类别名检索”流程不同，见解释.md 的版本边界。
        cls_x = self.fc_norm(t.mean(1))
        return self.head(cls_x)


class BEiT3ForCaptioning(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            **kwargs
    ):
        super(BEiT3ForCaptioning, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        # 将每个文字位置的隐藏向量映射到词表分数；此处虽叫 mlm_head，
        # 但 caption 生成还依赖下方的因果注意力 mask。
        self.mlm_head = nn.Linear(embed_dim, args.vocab_size)
        self.mlm_head.apply(self._init_weights)

    def forward(self, image, text_ids, padding_mask, language_masked_pos, text_len=None, incremental_state=None, **kwargs):
        # text_ids 为 [B,T] 整数词编号，text_len=T；增量解码可显式给不同于本次短输入的历史长度。
        text_len = text_len if text_len is not None else text_ids.size(1)
        # image_len=I，视觉小块数+视觉 CLS；不是像素 H×W，也不是 RGB 通道数。
        image_len = self.beit3.vision_embed.num_position_embeddings()
        max_len = text_len + image_len
        # uni_mask 名称沿用 UniLM 风格的统一注意力掩码；max_len=I+T，矩阵是 query×key，不含 batch 轴。
        # zeros 初始不允许任何连接，下方逐块填 1 允许，再反转为骨干需要的屏蔽约定。
        uni_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=text_ids.device)
        i_start, i_end = 0, image_len
        t_start, t_end = image_len, max_len
        # triangle mask for caption to caption
        # 文本只能看自己及更早的词，不能看到未来正确答案。
        # tril=triangular lower（下三角），包括主对角线：词 t 可读自己和左边的词，不能读右边答案。
        # 图片在前、文字在后，所以这里只赋值右下角的文字→文字子矩阵。
        uni_mask[t_start:t_end, t_start:t_end] = torch.tril(torch.ones(text_len, text_len, dtype=torch.long, device=text_ids.device))
        # full attention for caption to image
        # 生成任何词时都允许读取全部图像 token。
        uni_mask[t_start:t_end, i_start:i_end] = 1
        # full attention for image to image
        uni_mask[i_start:i_end, i_start:i_end] = 1
        # 上面 1 表示允许关注；骨干接口需要 1 表示屏蔽，因此取反。
        # 左上=图片互看，左下=文字看图片，右下=文字因果，右上=图片不能看文字。
        # 反转后 1=屏蔽、0=允许；不同注意力接口 mask 的 True 含义不一定相同，不能凭名字猜。
        uni_mask = 1-uni_mask

        # 【增量生成详解】None 表示无增量缓存；空字典{}也会进入这里。初始生成传{}，训练常传None，两者不可混淆。
        if incremental_state is not None:
            # 【增量生成详解】get_num_layers返回骨干层数；逐层建立缓存字典。idx=layer index，层编号，不是文字词编号。
            for idx in range(self.get_num_layers()):
                # 【增量生成详解】若这一层还没有缓存槽才初始化，已有层状态不能每轮清空，否则历史丢失。
                if idx not in incremental_state:
                    # 【增量生成详解】当前层建立空状态；注意力模块负责填入prev_key/prev_value。
                    incremental_state[idx] = {}
        
        # for incremental decoding
        # 【增量生成详解】positions 是文字位置编号，默认由文字嵌入自动生成；后续短输入必须显式指定历史位置。
        positions = None
        # 增量解码后续步骤会复用缓存，可能只输入新词而不重复传整张图。
        # 【增量生成详解】只有后续增量轮次才不传图片，前面缓存已有视觉token；若第一轮也无图，不能凭此接口宣称图像条件生成有效。
        if image is None:
            # 【增量生成详解】取注意力掩码最后两条query行，形状由[I+T,I+T]变为[2,I+T]；key轴保留完整视觉/文字历史，query只对应当前词和MASK。
            uni_mask = uni_mask[-2:]
            # 【增量生成详解】后续短输入不再传原padding mask；当前handler使用全有效短token，缓存长度由生成流程管理。
            padding_mask = None
            # start position (2 (fairseq starts at 2) + cur_position) is equal to text_len
            # 【增量生成详解】arange(start,end)左闭右开，当前输入2词时得到[text_len,text_len+1]；long转整数，unsqueeze(0)成为[1,2]，在batch共享。位置表从2开始，因此逻辑长度T对应实际历史词的位置偏移。
            positions = torch.arange(text_len, text_ids.size(1) + text_len, device=text_ids.device).long().unsqueeze(0)

        # 【增量生成详解】调用真实BEiT3骨干，传视觉/文字输入、统一注意力掩码与逐层缓存；outputs为含encoder_out的字典，不是句子。
        outputs = self.beit3(
            textual_tokens=text_ids, 
            visual_tokens=image, 
            text_padding_position=padding_mask,
            attn_mask=uni_mask,
            # 【增量生成详解】将缓存传入骨干，注意力逐层更新它；Handler之后还会按父beam行号重排，并删掉临时MASK缓存。
            incremental_state=incremental_state,
            # 【增量生成详解】显式位置编号只用于后续文字短输入，不能用0/1替代越来越大的历史位置，否则位置编码错位。
            positions=positions,
        )
        if image is not None:
            # feats=features（特征）；从第 I 项以后取文字隐藏状态 [B,T,D]，丢弃视觉 token 输出。
            text_feats = outputs["encoder_out"][:, image_len:]
        else:
            # 【增量生成详解】后续增量返回当前两个文字query的隐藏状态[BK,2,D]；没有重复输出整个视觉+历史序列。
            text_feats = outputs["encoder_out"]

        # 训练时可只在指定的文字目标位置计算词表预测。
        if language_masked_pos is not None:
            # language_masked_pos=[B,T] 指定需预测的位置；bool 转布尔，布尔索引合并样本/位置轴为 [M,D]。
            # M 是这个 batch 选中的文字位置总数；词表头输出 [M,V]，标签也必须是相同顺序的 M 个编号。
            text_feats = text_feats[language_masked_pos.bool()]

        # MLM=masked language modeling（遮盖词预测）；head 将 D 维隐藏状态变为 V 个词表分数。
        # 返回二元 tuple：logits 与缓存字典；并没有直接 decode 成一句话。
        return self.mlm_head(text_feats), incremental_state


class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]
        # VQA=visual question answering（视觉问答）；Pooler 取拼接序列第 0 个视觉 CLS 的融合状态。
        # head 输出常见答案候选的打分 [B,3129]，不是 BLIP 那种可自由生成任意词序列的解码器。
        cls_rep = self.pooler(x)
        return self.head(cls_rep)


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(
            self, 
            args,
            **kwargs
    ):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        # head=任务投影头；language/vision 各自一个 [D→D] 线性映射，让两模态进入可比较的向量空间。
        # bias=False 不加偏置；两边输出宽度相同不代表两个头共享参数。
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        # 检索使用 CLIP 式双向对比损失，而非对每个候选做词生成。
        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(), 
            world_size=utils.get_world_size(), 
        )
        # ones([]) 是零维标量；Parameter 可学习。存 log(1/τ)，τ=温度，初始化 0.07。
        # 用指数还原正尺度，点积再乘它；不是把向量直接减去温度。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        # 两边可以分开编码：评估时可缓存图片和文字向量，不必逐对运行骨干。
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None, 
                visual_tokens=image, 
                text_padding_position=None, 
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            # 单位长度化后，图文向量点积可比较余弦相似度。
            # 最后一维归一化，例 [3,4]→[0.6,0.8]；图文都单位长度后，点积等于余弦相似度。
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description, 
                visual_tokens=None, 
                text_padding_position=padding_mask, 
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None
        
        # only_infer=True 只拿向量；训练时则继续计算图文对比损失。
        if only_infer:
            return vision_cls, language_cls
        else:
            loss, logits_per_image, logits_per_text = self.criterion(
                # .exp() 把保存的对数尺度变成 1/τ；criterion 在训练时需要两种模态都存在。
                vision_cls, language_cls, self.logit_scale.exp())
            return loss, vision_cls, language_cls


@register_model
def beit3_base_patch16_224_imageclassification(pretrained=False, **kwargs):
    # 函数名分解：Base 规模，patch16=16 像素小块，224=图像边长，imageclassification=图像分类任务。
    # pretrained 参数本函数未自动下载权重；实际 checkpoint 加载在训练入口中处理，不要以名字推断已加载。
    args = _get_base_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=1000, **kwargs)
    return model


@register_model
def beit3_large_patch16_224_imageclassification(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=1000, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_nlvr2(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForVisualReasoning(args, num_classes=2, **kwargs)
    return model


@register_model
def beit3_large_patch16_224_nlvr2(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    model = BEiT3ForVisualReasoning(args, num_classes=2, **kwargs)
    return model


@register_model
def beit3_base_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_768_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=768, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_captioning(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForCaptioning(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_captioning(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    model = BEiT3ForCaptioning(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_captioning(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    model = BEiT3ForCaptioning(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model
