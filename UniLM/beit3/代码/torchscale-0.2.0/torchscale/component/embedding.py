# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# torch 提供 zeros/cat/arange 等张量操作；nn 管理有参数层；functional（缩写 F）提供函数式 embedding。
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLanguageEmbedding(nn.Module):
    def __init__(self, text_embed, vision_embed):
        super().__init__()
        self.text_embed = text_embed
        self.vision_embed = vision_embed

    def forward(self, textual_tokens, visual_tokens, **kwargs):
        if textual_tokens is None:
            return self.vision_embed(visual_tokens)

        if visual_tokens is None:
            return self.text_embed(textual_tokens)

        x1 = self.vision_embed(visual_tokens)
        x2 = self.text_embed(textual_tokens)

        return torch.cat([x1, x2], dim=1)


class VisionEmbedding(nn.Module):
    # Vision=视觉，Embedding=嵌入：把连续像素转换为 Transformer 的向量序列。
    # 本模块不是 BEiT v2 用来产生离散训练目标的视觉 tokenizer，两个角色必须分清。
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        contain_mask_token=False,
        prepend_cls_token=False,
    ):
        super().__init__()
        # 单整数边长转换成 (高,宽) tuple，以便统一处理图像两个空间轴；这里实际构造正方形。
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        # // 是整数整除；num_patches 为横向小块数×纵向小块数，不包含稍后添加的 CLS。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # proj=projection（投影）。卷积核与步长都等于 patch 大小，每个互不重叠 patch 得到 embed_dim 个数。
        # Conv2d 输入 [B,C,H,W]，输出 [B,D,H/p,W/p]；in_chans 通常为 3 个 RGB 通道。
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if contain_mask_token:
            # MASK token 是一个可学习替代向量 [1,1,D]，覆盖被遮盖 patch，不是编号 -100 或纯黑图块。
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            # CLS=classification 汇总标记，在视觉序列开头；它是参数，不是从原图截取的一块。
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        # 返回当前位置表需要的序列长度；开了视觉 CLS 时为 patch 数+1。
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        # 四元解包：B=batch（样本数），C=channel（通道），H=height，W=width。
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 卷积 [B,D,Gh,Gw]；flatten(2) 合并第 2 维起的网格→[B,D,P]；transpose(1,2)→[B,P,D]。
        # 这些 API 都不是将像素转成整数词编号，输出仍是连续浮点特征。
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            # expand 广播 [1,1,D] 到 [B,P,D]，-1 表示该轴保持 D；所有遮盖位置共用同一个可学习向量。
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            # masked_position=[B,P]，True=要遮盖；unsqueeze(-1) 加通道轴→[B,P,1]。
            # type_as 把布尔值转成与浮点 mask_token 相同的类型，得到 0/1 权重，供广播乘法。
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            # w=0 保留原 patch；w=1 使用 MASK 向量。乘法对每元素进行，不是矩阵乘法 @。
            # 本行仍允许梯度回到可学习 MASK 参数，不是用 detach 去掉学习关系。
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            # CLS 在前，patch 在后；cat 沿序列轴 dim=1，输出 [B,P+1,D]。
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class TextEmbedding(nn.Embedding):
    # 继承 PyTorch Embedding 的整数查表功能，只改初始化；输入 [B,T] 得到 [B,T,D]。
    def reset_parameters(self):
        # normal_ 就地填正态随机数；D**-0.5 = 1/sqrt(D)，用隐藏宽度调整初始标准差。
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


class PositionalEmbedding(nn.Embedding):
    # Positional=位置：按序列位置查另一张表，不是再次按内容词编号查 TextEmbedding。
    def forward(
        self,
        x,
        positions=None,
        **kwargs,
    ):
        if positions is None:
            # being consistent with Fairseq, which starts from 2.
            positions = (
                # arange(start,stop) 不含 stop；长度 T 生成 2..T+1，共 T 项。
                # long=int64 适合查表；unsqueeze(0) 得 [1,T]，同一位置序列可在 batch 上广播。
                torch.arange(2, x.size(1) + 2, device=x.device).long().unsqueeze(0)
            )
        # F.embedding 是函数式查表，self.weight 是 [位置容量,D]；positions 为 [1,T]→[1,T,D]。
        # 后续加到 [B,T,D] 内容表示时在 batch 维广播；不是把位置整数直接加到词编号上。
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
