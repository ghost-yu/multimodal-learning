# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# torch 是 PyTorch 主库，提供多维 Tensor、拼接、设备与自动求导功能。
# 本文件常用 cat 拼接序列、zeros 创建掩码；Tensor.size/shape 取得维度。
import torch
# nn=neural network（神经网络），Module 是模型基类，自动登记子模块及其可学习参数。
import torch.nn as nn

# torchscale 是本次下载的 0.2.0 依赖源码；Encoder 是 Transformer 编码骨干，不是 tokenizer。
# 它处理已嵌入的 token 序列，自注意力共享，FFN/部分归一化通过 MultiwayWrapper 分成 A/B。
from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    VisionEmbedding,
)
# Mutliway 是这份源代码保留的拼写（通常英语应写 Multiway）；不要自行改标识符导致导入失败。
# 这里为视觉和文字提供不同的位置嵌入表，沿序列维 dim=1 路由；不是三专家门控网络。
from torchscale.component.multiway_network import MutliwayEmbedding


class BEiT3(nn.Module):
    # BEiT 名字源于 BERT Pre-Training of Image Transformers；3 表示这一代多模态模型。
    # BERT=Bidirectional Encoder Representations from Transformers，强调双向上下文编码。
    def __init__(self, args, **kwargs):
        # self 是当前对象；args 是配置对象，可用 args.encoder_embed_dim 读取属性。
        # **kwargs 接收多余关键字参数；这里没有使用它们，不意味着可用它们覆盖任意配置。
        super().__init__()
        # super 调父类初始化，使后续 self.text_embed 等模块能正确登记。
        self.args = args
        # assert 表达前置条件；失败抛 AssertionError。multiway 必须启用，词表必须有正数量的词。
        assert args.multiway
        assert args.vocab_size > 0
        # 当前实现不接受“输入词嵌入和输出词分类权重共享”的配置。
        assert not args.share_encoder_input_output_embed
        # TextEmbedding 为整数词编号查表：[B,T]→[B,T,D]；vocab_size=V，encoder_embed_dim=D。
        # 它不是把 Python 字符串直接转成向量，字符串需先经过 SentencePiece 分词。
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        # VisionEmbedding 将 RGB 像素 [B,3,H,W] 经 patch 卷积变 [B,P,D]，并在前面加视觉 CLS。
        # img=image，chans=channels（通道），embed=embedding（嵌入）；mask_token 是被遮盖 patch 的替代向量。
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            # contain=包含；启用可学习的视觉 MASK token，用于图像位置被遮盖时替换 patch 向量。
            contain_mask_token=True,
            # prepend=加在开头；CLS=classification 汇总标记，视觉序列第 0 项。
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        # embed_positions 是位置查表器，给不同位置加入不同向量，使注意力能分辨顺序/位置。
        # 两个 module 按顺序作为 A=视觉、B=文字，并非输入编号共用一张位置表。
        embed_positions = MutliwayEmbedding(
            modules=[
                # num_position_embeddings 包含视觉 CLS；+2 保留 Fairseq 从编号 2 起使用位置的约定。
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                # source=输入序列；max_source_positions 是文字位置表容量，不是本次 batch 的实际文字长度。
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            # dim=dimension（轴），[B,L,D] 的第 1 维是 token 序列，将按视觉/文字边界切分。
            dim=1,
        )
        # 编码器接收下方先算好的 token_embeddings，因此内部 embed_tokens=None，避免重复查表。
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            # 不在骨干内部投到最终任务标签；分类/检索/caption 由 modeling_finetune.py 的头负责。
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        # textual_tokens 是 [B,T] 整数文字编号；visual_tokens 虽然叫 tokens，实际传入的是 RGB 图片。
        # or=或者：至少一个模态存在。None 表示没提供，和全零 Tensor 不是同一回事。
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            # 纯图片路径：x=[B,P+1,D]，+1 为开头视觉 CLS；可选 masked_position 标出遮盖 patch。
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            # split_position=序列切分边界；-1 是本实现特殊约定“全部走 A（视觉）”。
            # 这里不是 Python 切片的最后一项，路由类明确检查 ==-1。
            multiway_split_position = -1
        elif visual_tokens is None:
            # 纯文字路径：x=[B,T,D]，文字开始标记由数据 tokenizer 提供，本函数没有额外添加文字 CLS。
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            # 0 是特殊约定“全部走 B（文字）”，不是视觉序列里 0 个元素后仍调用 A。
            multiway_split_position = 0
        else:
            # 混合路径先编码图片：x1=[B,P+1,D]；x1/x2 是中间局部名字，应记住各自模态。
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            # 图片在前、文字在后；边界是视觉 token 数 P+1，包括视觉 CLS。
            multiway_split_position = x1.size(1)
            # x2 是文字嵌入 [B,T,D]，和 x1 宽度同为 D，才能沿 token 维拼接。
            x2 = self.text_embed(textual_tokens)
            # cat=concatenate（拼接），得到 [B,P+1+T,D]；顺序对专家切分与任务取 CLS 至关重要。
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        # shape[:-1] 取 [B,P+1]；zeros→to→bool 生成全 False 的视觉 padding 标记。
                        # 此接口 True 表示补齐、应屏蔽，和 CoCa 的 text_valid=True 表示真词恰好相反。
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        # encoder_out 是字典，不是单独的向量；encoder_out['encoder_out'] 才是隐藏状态 Tensor。
        # src_tokens=None：已传 token_embeddings=x，因此不用编码器再次从编号生成嵌入。
        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            # attn_mask 控制 query/key 之间哪些位置可互看；padding_mask 只描述哪些 token 无效，职责不同。
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            # incremental=增量，state=状态；生成任务可存各层历史 K/V，减少重复处理前缀。
            incremental_state=incremental_state,
            positions=positions,
        )
        # 把边界随结果一起返回，融合任务可按它定位文字第一项；这个整数不参与求导。
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out
