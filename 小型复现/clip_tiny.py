"""CLIP 教学模型：图片和文字各自编码，最后比较两个全局向量。"""

# math 提供对数，初始化 CLIP 的可学习温度参数。
import math

# PyTorch 提供张量、自动求导和模型基类。
import torch
# nn 包含卷积、词向量、Transformer 和线性投影。
from torch import nn
# F.normalize 把向量缩放到长度为 1。
from torch.nn import functional as F


class TinyCLIP(nn.Module):
    """保留 CLIP 的双编码器、归一化相似度和可学习温度。"""

    def __init__(self, vocab_size: int, image_size: int = 64, patch_size: int = 8, text_len: int = 20, width: int = 64, embed_dim: int = 64):
        # 初始化父类，PyTorch 才能追踪下面定义的可训练参数。
        super().__init__()
        # 图片宽高必须能被 patch 大小整除，否则边缘像素会被丢弃。
        if image_size % patch_size != 0:
            # 在创建模型时直接报告尺寸错误。
            raise ValueError("image_size 必须能被 patch_size 整除")
        # 64×64 图片按 8×8 切块，会得到 8×8=64 个图像 token。
        patch_count = (image_size // patch_size) ** 2
        # 卷积核和步长都等于 patch 大小：每块 RGB 像素映射成 width 维向量。
        self.image_patch = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        # 图像 CLS 是额外的可学习向量，用来汇总全部图像块。
        self.image_cls = nn.Parameter(torch.zeros(1, 1, width))
        # 图像位置向量告诉 Transformer 每块原来位于图中的哪里。
        self.image_position = nn.Parameter(torch.randn(1, patch_count + 1, width) * 0.02)
        # 一层图像 Transformer，输入维度顺序是 [批量, token 数, width]。
        image_layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True)
        # CLIP 两边分别编码；这个实例只用于图片。
        self.image_encoder = nn.TransformerEncoder(image_layer, num_layers=1, enable_nested_tensor=False)
        # 图像 CLS 的 width 维表示投影到统一的 embed_dim 维空间。
        self.image_projection = nn.Linear(width, embed_dim, bias=False)
        # 文字整数 ID 查表得到词向量；编号 0 是 PAD，始终对应零向量。
        self.word_embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        # 每个文字位置也有自己的可学习位置向量。
        self.text_position = nn.Parameter(torch.randn(1, text_len, width) * 0.02)
        # 文字 Transformer 与图像 Transformer 结构相似，但参数完全独立。
        text_layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True)
        # 文字在自己的编码器中处理，直到最终相似度计算前都不读取图片。
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=1, enable_nested_tensor=False)
        # 用结束符 SEP 的输出表示整句话，再投影到与图像相同的维度。
        self.text_projection = nn.Linear(width, embed_dim, bias=False)
        # CLIP 把温度的倒数取对数存为参数；exp 后得到正的相似度缩放因子。
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """图片 [B,3,64,64] → 单位长度的图像向量 [B,E]。"""
        # 卷积后是 [B,width,8,8] 的图像网格。
        patches = self.image_patch(images)
        # flatten(2) 合并两个网格维，再 transpose 成 [B,64,width]。
        patches = patches.flatten(2).transpose(1, 2)
        # 同一个可学习 CLS 参数扩展到 B 张图片，但不会真正复制 B 份权重。
        cls = self.image_cls.expand(images.shape[0], -1, -1)
        # 在 token 维拼接 CLS 与 64 个图像块，得到 [B,65,width]。
        tokens = torch.cat((cls, patches), dim=1)
        # 加位置向量后由图像 Transformer 让各块交换信息。
        encoded = self.image_encoder(tokens + self.image_position)
        # 第 0 个位置是汇总整图的 CLS，再投影到共同的检索空间。
        projected = self.image_projection(encoded[:, 0])
        # 每个向量除以自己的 L2 长度，使点积等于余弦相似度。
        return F.normalize(projected, dim=-1)

    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """文字 ID [B,T] → 单位长度的句子向量 [B,E]。"""
        # 先查词向量并加入位置信息，形状都是 [B,T,width]。
        words = self.word_embedding(texts) + self.text_position
        # T 是补齐后的句子长度，本项目默认 20。
        length = texts.shape[1]
        # 主对角线上方为 True，禁止某个词读取它后面的词；这是因果注意力。
        future_mask = torch.ones(length, length, dtype=torch.bool, device=texts.device).triu(diagonal=1)
        # PAD=0 是为了对齐长度填充的空位，不能当作真实单词参与注意力。
        padding_mask = texts.eq(0)
        # 文字仅在自己的 Transformer 中编码，得到每个词位置的特征。
        encoded = self.text_encoder(words, mask=future_mask, src_key_padding_mask=padding_mask)
        # 本数据处理器保证每句话有 SEP=3；eq 形成 True/False 位置表。
        sep_positions = texts.eq(3).int().argmax(dim=1)
        # arange 产生每句话的批量编号；与 sep_positions 合起来取每句的 SEP 向量。
        sentence_vectors = encoded[torch.arange(texts.shape[0], device=texts.device), sep_positions]
        # 投影到与图像相同的维度，再缩放为单位长度。
        return F.normalize(self.text_projection(sentence_vectors), dim=-1)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """B 对图文 → [B,B] 分数矩阵，正确图文位于对角线。"""
        # 图片各自得到一个 [E] 向量，拼起来是 [B,E]。
        image_vectors = self.encode_image(images)
        # 文字各自得到一个 [E] 向量，拼起来也是 [B,E]。
        text_vectors = self.encode_text(texts)
        # exp 保证缩放因子为正；限制上界避免小数据训练时数值失控。
        scale = self.logit_scale.exp().clamp(max=100)
        # [B,E] 乘 [E,B] 得到 [B,B]：第 i 行第 j 列表示图 i 与句 j 的分数。
        return scale * image_vectors @ text_vectors.T
