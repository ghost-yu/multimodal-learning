# 教学副本：保持官方仓库现有可执行语句不变，只增补给初学者的中文解释。
# 结构对照：原始文件 BLIP/models/vit.py；建议同时阅读 项目讲解/BLIP-从零重讲 系列。
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on timm code base
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
'''

# torch 是 PyTorch 主模块：张量、矩阵乘法、拼接、初始化和插值都依赖它。
import torch
# nn = neural network；Module 是模型基类，Linear/LayerNorm/Dropout/Parameter 是本文件主要层。
import torch.nn as nn
# F = functional（函数式接口）；这是从原仓库保留的导入，ViT 主前向主要调用 nn 层。
import torch.nn.functional as F
# partial 把函数/构造器的某些参数提前固定，例如固定 LayerNorm 的 eps。
from functools import partial

# PatchEmbed 通常用卷积将图片切成不重叠 patch，并投影到 token 维度。
# timm 的 PatchEmbed 把像素图切成小块并投影为 token；_cfg 是模型配置辅助项。
from timm.models.vision_transformer import _cfg, PatchEmbed
# register_model 是 timm 注册模型用的装饰器；BLIP 此处直接构造 VisionTransformer。
from timm.models.registry import register_model
# trunc_normal_ 原地初始化权重；DropPath 在训练时随机跳过整条残差分支。
from timm.models.layers import trunc_normal_, DropPath
# named_apply 递归应用带名字的函数；adapt_input_conv 调整旧卷积权重的输入通道。
from timm.models.helpers import named_apply, adapt_input_conv

# fairscale 的梯度检查点：反向时重算部分前向，花额外计算来降低显存占用。
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 未指定输出维度时，输出仍等于输入维度，便于做残差相加。
        out_features = out_features or in_features
        # hidden_features 是 MLP 中间层宽度；未提供时用输入宽度 in_features。
        hidden_features = hidden_features or in_features
        # 两个线性层中间扩宽维度，再压回原维度。
        # fc=fully connected（全连接）；先将每个 token 的特征从输入宽度扩到隐藏宽度。
        self.fc1 = nn.Linear(in_features, hidden_features)
        # act_layer 默认是 nn.GELU 这个类；加括号创建可调用的激活层对象。
        self.act = act_layer()
        # 第二层把扩宽后的 token 特征压回输出宽度，便于与残差 x 相加。
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout 在训练时随机置零部分数值；评估模式不丢弃，张量形状始终不变。
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 输入、输出都是 [B,N,D]，逐 token 变换，不在这里混合 token 之间的信息。
        # x 是 [B,N,D] token；Linear 仅改变最后的特征维，不改变 batch 或 token 数。
        x = self.fc1(x)
        # GELU 是非线性激活，否则连续线性层等效于一个线性层。
        # GELU 增加非线性，否则两个连续 Linear 可等效合成一个 Linear。
        x = self.act(x)
        # 本 MLP 中两次使用同一个 Dropout 层；训练时可能置零部分特征，形状不变。
        x = self.drop(x)
        # fc2 把中间层宽度压回输出宽度，使外面的残差相加仍能保持形状。
        x = self.fc2(x)
        # 本 MLP 中两次使用同一个 Dropout 层；训练时可能置零部分特征，形状不变。
        x = self.drop(x)
        # 返回本层处理后的 token 张量；调用处可根据位置区分 MLP、Attention 或完整 ViT。
        return x


class Attention(nn.Module):
    """多头自注意力：让同一图片的不同 patch token 互相读取信息。

    Q = Query（查询），K = Key（键），V = Value（值）；Q 与 K 算匹配分数，
    softmax 后得到权重，再按权重汇总 V。H 个头相当于从 H 组特征角度观察。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # num_heads 是注意力头数 H；后面把总宽度 C 均分为 H 个较小特征组。
        self.num_heads = num_heads
        # 每个头分到 D/H 维；例如 768/12=64。
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # 点积除以 sqrt(head_dim)，避免维度大时 softmax 过于尖锐。
        # head_dim 是每个注意力头的维数；默认缩放系数 1/sqrt(head_dim)。
        self.scale = qk_scale or head_dim ** -0.5
        # 一次线性投影同时产生 Query、Key、Value 三组向量。
        # 一个 Linear 同时产生 Q/Query、K/Key、V/Value 三组向量，总宽度 3×dim。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 这个 Dropout 作用于 softmax 后的注意力权重。
        self.attn_drop = nn.Dropout(attn_drop)
        # 多个头合并后，输出再经过一次可训练线性投影。
        self.proj = nn.Linear(dim, dim)
        # 这个 Dropout 作用于输出投影后的特征；与 attn_drop 位置不同。
        self.proj_drop = nn.Dropout(proj_drop)
        # 两个属性只保存可视化需要的数据，None 表示当前没有记录。
        self.attn_gradients = None
        # 默认还没有保存可视化注意力图，常规前向不依赖这个属性。
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        # 被 register_hook 调用时保存当前注意力张量的反向梯度。
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        # 读取最近保存的注意力梯度供解释或可视化。
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        # 保存 softmax 后的 [B,H,N,N] 注意力概率供解释。
        self.attention_map = attention_map
        
    def get_attention_map(self):
        # 读取最近保存的注意力概率图。
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        # B=批量大小，N=token 数，C=总特征维度。
        # B 是图片数，N 是图像 token 数，C 是每个 token 的全部特征维度；x 已是 token 而非像素。
        B, N, C = x.shape
        # reshape 分出 Q/K/V 与多个头；permute 后是 [3,B,H,N,C/H]。
        # Linear 一次产生 Q/K/V；reshape 拆出三组和多个头；permute 后形状 [3,B,H,N,C/H]。
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 分别取得 Q、K、V，单个张量形状 [B,H,N,C/H]。
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # @ 是矩阵乘法；每个 Query 与全部 Key 点积，得到 [B,H,N,N]
        # 的两两分数。transpose(-2,-1) 只交换 Key 的最后两个维度。
        # K 的最后两维互换后，Q×Kᵀ 得 [B,H,N,N]；每个 token 给全部 token 打分。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对最后一维归一化：某个 token 读取所有 token 的权重之和为 1。
        # 每个查询位置沿全部被读取位置归一化，最后一维的权重和为 1。
        attn = attn.softmax(dim=-1)
        # attn 仍为 [B,H,N,N]；训练时随机置零部分权重以做正则化。
        attn = self.attn_drop(attn)
                
        # 仅在调用端指定要查看本层注意力时才登记保存函数。
        if register_hook:
            # 仅在可视化注意力时保存权重和反向梯度。
            self.save_attention_map(attn)
            # register_hook 在反向传播经过 attn 时调用保存梯度的函数。
            attn.register_hook(self.save_attn_gradients)        

        # 用权重汇总 V；合并 H 个头，恢复 [B,N,C]。
        # 按注意力权重加权 V，再把 H 个头并回 C 维，恢复 [B,N,C]。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 注意力头已经合并成 [B,N,C]，这里再次按 C 维做线性混合。
        x = self.proj(x)
        # 输出投影后做 Dropout，最终返回给外层 Block 的残差分支。
        x = self.proj_drop(x)
        # 返回本层处理后的 token 张量；调用处可根据位置区分 MLP、Attention 或完整 ViT。
        return x


class Block(nn.Module):
    """一个 ViT 块：注意力残差 + MLP 残差。"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        # 每条子路径先 LayerNorm，再做变换，再与旧 x 相加。
        self.norm1 = norm_layer(dim)
        # 创建本块自注意力：所有图像 token 相互读取，输出维度保持 D。
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # DropPath 训练时随机跳过整个残差分支；0 时直接原样通过。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个归一化层放在 MLP 前；norm1 则放在注意力前。
        self.norm2 = norm_layer(dim)
        # mlp_ratio=4 时，隐藏层通常从 768 扩到 3072。
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP 逐 token 改变特征；跨 token 的信息交换主要由上一条 attention 完成。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 只有显式启用梯度检查点时才包装注意力和 MLP，改变内存策略而非数学目标。
        if use_grad_checkpointing:
            # 梯度检查点以额外计算换显存，适合模型大时使用。
            self.attn = checkpoint_wrapper(self.attn)
            # 包装后仍像普通模块一样调用，但反向时会重新计算被省下的激活。
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        # 第一个残差：token 之间交流信息；维度仍是 [B,N,D]。
        # 从内到外读：归一化、token 间注意力、可能丢弃分支、与旧 x 相加（残差）。
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        # 第二个残差：对每个 token 的特征做非线性变换。
        # 第二条残差先归一化，再逐 token 通过两层 MLP；此时 x 已是上一句更新后的 x。
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 返回本层处理后的 token 张量；调用处可根据位置区分 MLP、Attention 或完整 ViT。
        return x

    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 use_grad_checkpointing=False, ckpt_layer=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        # 保存每个图像 token 的维度，例如 Base 的 768。
        # 两个属性都记录 token 特征宽度 D；num_features 是兼容其他视觉模型的命名。
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # 未指定 norm_layer 时，用 partial 建一个 eps 固定为 1e-6 的 LayerNorm 构造器。
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # 224/16=14，每行 14 块、每列 14 块，一共 196 个 patch。
        # PatchEmbed 将图片切成固定大小块并投影为 token；224/16 得到 14×14=196 块。
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # 这是图片切出的 patch 总数，不含随后额外加的 CLS。
        num_patches = self.patch_embed.num_patches

        # CLS 是额外的可训练 token，用于汇总整张图片的信息。
        # CLS 是额外的全局汇总 token；Parameter 会被优化器训练，初始形状 [1,1,D]。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 每个 patch 和 CLS 各有一组可训练的位置向量。
        # 每个 patch 和 CLS 各有一个位置向量；+1 是 CLS 的位置。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # patch 与 CLS 加位置向量后，才进入这个 Dropout。
        self.pos_drop = nn.Dropout(p=drop_rate)

        # dpr=drop path rates；linspace 均匀生成 depth 个概率，item 把标量张量转 Python 数。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # 按 depth 创建相同结构但参数各自独立的 Transformer 层。
        # ModuleList 像列表却会把每个 Block 的参数登记到模型中，普通列表不够。
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer)
            )
            for i in range(depth)])
        # 全部 Transformer 块后再归一化，仍保留所有 CLS/patch token。
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # Module.apply 递归访问模型内部各层，为 Linear/LayerNorm 执行初始化规则。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 初始化递归访问到 Linear 时，为其 weight/bias 分别填初值。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # 检查偏置是否存在，才可用 nn.init.constant_ 原地置零。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # LayerNorm 的 weight/bias 采取 1/0 初值，使初始不任意缩放特征。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回优化器可选择不做权重衰减的参数名集合；函数本身不修改优化器。
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1):
        """图片 [B,3,H,W] -> 图像 token [B,1+patch数,D]。"""
        # forward 的 x 在这一行仍是图片像素张量 [B,3,H,W]；B 只记录批量大小。
        B = x.shape[0]
        # PatchEmbed 把图像切块并映射为 [B,P,D]。
        # 这一行之后 x 改指 patch token [B,patch数,D]，不再是原始像素图。
        x = self.patch_embed(x)

        # 原 CLS 只有 [1,1,D]，expand 后让 batch 中每张图都有一个 CLS。
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # 沿 token 维拼接，CLS 放在第 0 位，形状变成 [B,P+1,D]。
        # 沿 token 维拼接，CLS 放第 0 位，所以之后 x[:,0,:] 是整图汇总位置。
        x = torch.cat((cls_tokens, x), dim=1)
  
        # 给每个位置加入位置信息，否则注意力不知道 patch 在图中的位置。
        # 按现有 token 数切出位置向量并逐元素相加；张量形状不变。
        x = x + self.pos_embed[:,:x.size(1),:]
        # 此时 x 已是加入位置向量的 [B,P+1,D]，训练期按概率做 dropout。
        x = self.pos_drop(x)

        # 依次经过 12 或 24 个 Transformer block。
        # enumerate 同时给层号 i 和本层对象 blk；每层输出继续赋给 x，层层更新。
        for i,blk in enumerate(self.blocks):
            # 第 i 层接上一层输出；比较式为 True 时记录该层注意力图。
            x = blk(x, register_blk==i)
        # 最后一次 LayerNorm 不改变 [B,P+1,D] 的形状，随后返回全部 token。
        x = self.norm(x)
        
        # BLIP 保留全部 patch 特征，供文字模型做图文交叉注意力。
        # 返回本层处理后的 token 张量；调用处可根据位置区分 MLP、Attention 或完整 ViT。
        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        # checkpoint_path 是 .npz 权重文件路径；prefix 是文件中旧参数键名的公共前缀。
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    # NumPy 负责读取 Google/Flax 格式 .npz 权重；这条路径不是日常 ViT 前向。
    import numpy as np

    def _n2p(w, t=True):
        # n2p 可读作 NumPy-to-PyTorch；w 是旧数组，t 控制是否按框架布局转置。
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            # 若 Flax 数组是 1×1×1×D 之类退化形状，flatten 压为一维以匹配 PyTorch 参数。
            w = w.flatten()
        # t 是 _n2p 的转置开关；框架之间线性和卷积权重轴顺序可能不同。
        if t:
            # 四维卷积核需把 Flax 布局转为 PyTorch [输出通道,输入通道,高,宽]。
            if w.ndim == 4:
                # 明确交换四个轴，保证旧卷积核能写入 PyTorch Conv2d.weight。
                w = w.transpose([3, 2, 0, 1])
            # 三维位置或其他权重按各自旧/新框架布局重新排序。
            elif w.ndim == 3:
                # 把三维数组的轴 2、0、1 依次放到结果的第 0、1、2 维。
                w = w.transpose([2, 0, 1])
            # 二维 Linear 矩阵在不同框架常互为转置。
            elif w.ndim == 2:
                # 二维权重的行列互换，旧输入/输出通道与 PyTorch 线性层对应。
                w = w.transpose([1, 0])
        # 转换为 PyTorch 张量，供下面的 copy_() 写入已创建好的模型参数。
        return torch.from_numpy(w)

    # w 是可按字符串键读取的 .npz 对象，每个键对应一层旧权重数组。
    w = np.load(checkpoint_path)
    # 若旧 Flax 键名有 opt/target/ 前缀而调用者未给 prefix，则自动补上。
    if not prefix and 'opt/target/embedding/kernel' in w:
        # npz 键名若带该公共前缀，后面读取所有层时统一加在键名前。
        prefix = 'opt/target/'

    # hasattr 查询模型是否有 CNN backbone；标准 BLIP 的纯 ViT 切块通常走 else。
    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        # 兼容某些 CNN backbone 结构；纯 BLIP ViT 通常不会进入该 hybrid 分支。
        stem_only = not hasattr(backbone, 'stem')
        # 三元表达式选择 CNN 根模块或其 stem 子模块，后面复制卷积和归一化权重。
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        # 有完整 CNN stem 时继续逐 stage 复制混合视觉骨干的参数。
        if not stem_only:
            # 遍历 CNN 的第 i 个 stage；这是混合视觉骨干的兼容权重路径。
            for i, stage in enumerate(backbone.stages):
                # 再遍历该 stage 的第 j 个卷积块，拼出旧检查点的层键。
                for j, block in enumerate(stage.blocks):
                    # f-string 根据 i/j 生成旧 Flax checkpoint 中从 1 开始编号的键名前缀。
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    # 此混合 CNN 块含三个卷积/归一化子层；r 依次取 0、1、2。
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    # 有跳接下采样层时，卷积和归一化权重也要从旧 checkpoint 复制。
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        # 先把旧 patch 嵌入卷积核转成 PyTorch 张量；变量名 w 表权重。
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        # 将旧 patch 卷积核输入通道适配当前图片通道，再复制到模型。
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    # copy_ 只把数值写进现有参数；没有新建层，末尾 _ 表示原地修改。
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    # 旧位置向量随旧 patch 网格长度而定；图片分辨率改动时长度可能不匹配。
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    # 旧/新网格不同才尝试重采样；该旧 .npz 辅助分支的 resize_pos_embed 需另行提供。
    if pos_embed_w.shape != model.pos_embed.shape:
        # 原仓库此兼容分支引用 resize_pos_embed，但本文件未定义/导入它；
        # 若用这条 .npz 路径且形状不同，需要先补全依赖。常规 BLIP checkpoint
        # 加载走 models/blip.py 的 interpolate_pos_embed，不经过这里。
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    # enumerate 给每个 Transformer 层编号 i，逐层把 Flax 权重移入 PyTorch Block。
    for i, block in enumerate(model.blocks.children()):
        # f-string 把当前块编号 i 插入旧检查点键名，供后续 norm/QKV/MLP 查权重。
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        # mha=multi-head attention；拼旧权重文件里注意力层的公共键前缀。
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        # 旧文件分别存 Q/K/V；当前 Attention 用一个 qkv Linear，故按 Q,K,V 顺序拼权重。
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        # 偏置也必须按完全相同顺序拼，否则 forward 中 qkv[0:3] 会拿错含义。
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        # MLP 有 fc1/fc2 两层；r=0 对应第一层，r=1 对应第二层。
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    """当输入分辨率改变时，按二维网格插值旧检查点的 patch 位置向量。"""
    # 位置向量最后一维是模型隐藏维度 D，例如 768。
    embedding_size = pos_embed_checkpoint.shape[-1]
    # 新 ViT 的 patch 数由目标图片大小 / patch_size 决定。
    num_patches = visual_encoder.patch_embed.num_patches
    # 除 patch 外保留 CLS 等特殊 token 的数量。
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    # 旧 patch 网格假设为正方形，对 patch 数开平方得到每边格子数。
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    # 网格尺寸相同无需修改；不同则只插值 patch 位置，特殊 token 保持原样。
    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # reshape 为旧二维网格；permute 改成插值函数要求的 [B,D,H,W]。
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        # 双三次插值把旧网格变成新网格，例如 14×14 -> 24×24。
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # 再变回 [B,新patch数,D]，与模型普通 token 布局一致。
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # 把未改动的 CLS 位置向量接在最前面。
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        # 新位置向量由未插值的特殊 token 加已插值的 patch token 拼成，形状与目标 ViT 一致。
        return new_pos_embed    
    else:
        # 旧/新网格边长一样时无需插值，直接返回原位置向量。
        return pos_embed_checkpoint
