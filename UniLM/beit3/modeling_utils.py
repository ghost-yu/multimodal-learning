# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

# math 是 Python 标准数学库；sqrt(x) 求平方根，用于按网络深度缩放初始化权重。
import math
# torch 是 PyTorch 主库，提供 Tensor、自动求导与 jit；这里的 jit.ignore 是跳过脚本编译的装饰器。
import torch
# nn=neural network：Module 管理参数，Linear 做线性投影，LayerNorm 做每 token 的特征归一化。
import torch.nn as nn
# timm 是图像模型工具库，官方 requirements 锁为 0.4.12；不能随便用新版视为完全兼容。
# trunc_normal_=截断正态初始化，尾部 _ 表示就地修改张量；as 为导入对象另取名字，避免与下方包装函数重名。
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

# BEiT3 骨干来自外部 torchscale，而不是这个目录本身；要求版本 0.2.0 的源码现保存在 代码/ 下供阅读。
# 普通 import 使用当前 Python 环境的安装包，不会因为下载到 代码/ 就自动使用它。
from torchscale.model.BEiT3 import BEiT3
# EncoderConfig 是属性式配置对象；encoder_embed_dim=隐藏宽度，encoder_ffn_embed_dim=FFN 中间宽度。
from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    # tensor 为待初始化权重；mean=平均值，std=standard deviation（标准差）。
    # a/b 指截断范围，这个包装显式使用 [-std,std]，不能直接描述成默认的 [-2,2]。
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    # Base=基础规模，使用 12 层、宽度 768、12 个头；cfg/config 都是 configuration 的简称。
    # img_size=图边长；patch_size=小块边长；vocab_size=词表数 V；mlp_ratio=FFN 中间宽度倍率。
    # multiway=True 开启模态分支，FFN 与部分归一化分为 A/B；共享注意力不是给两路复制两套。
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    # Large=更大规模：24 层、宽度 1024、16 个头；不是论文 Giant 的 40 层、宽度 1408。
    # checkpoint_activations 表示激活检查点重算以节省显存，不是 checkpoint 权重文件路径。
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3Wrapper(nn.Module):
    # Wrapper=包装器，把共享骨干与下游任务头连接起来；不同任务子类继承它。
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        # args 是配置对象；self.beit3 保存真正骨干，使 forward 可调用它并读取 encoder_out。
        self.beit3 = BEiT3(args)
        # apply(fn) 递归遍历当前对象与所有子模块，逐个执行初始化；不是 Tensor.apply 的逐元素运算。
        self.apply(self._init_weights)

    def fix_init_weight(self):
        # fix=调整，init=initialization（初始化）。此遗留辅助方法访问 self.blocks，当前基类未定义该属性。
        # 不要未核查就直接调用；实际骨干的层在 self.beit3.encoder.layers，接口并非这里写的 blocks/attn/mlp。
        def rescale(param, layer_id):
            # div_ 就地除法；第 l 层输出投影除以 sqrt(2l)，控制深层网络初始残差幅度。
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        # get=取得，num=number（数量）；返回骨干编码层数，caption 缓存按这个数量初始化。
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回优化器应排除权重衰减的参数名集合；set 是无序且去重的容器，不是参数值本身。
        # pos_embed=位置嵌入、cls_token=汇总标记、logit_scale=检索分数尺度。
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        # m=module，是 apply 遍历时传来的一个子模块；不是整个 batch，也不是遮盖 mask。
        if isinstance(m, nn.Linear):
            # Linear 的 weight 保存投影矩阵，以小截断正态初始化；已有 bias 时将其设为 0。
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            # LayerNorm 的 weight 是缩放系数而非线性矩阵；置 1、bias 置 0，从标准归一化开始学习。
            nn.init.constant_(m.weight, 1.0)
