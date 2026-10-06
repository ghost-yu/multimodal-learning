""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
# partial 用来预先指定 LayerNorm 的 eps 等参数。
from functools import partial

# torch 是张量计算主体，nn 定义网络层，F 提供函数式线性层。
import torch
import torch.nn as nn
import torch.nn.functional as F

# DropPath 随机跳过残差分支；to_2tuple 把一个整数变成高宽两个整数。
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 注册模型名，之后 VLMo 可用 timm.create_model("vlmo_base_patch16") 创建。
from timm.models.registry import register_model
# 多 GPU 时只让主进程打印信息。
from pytorch_lightning.utilities.distributed import rank_zero_info


class Mlp(nn.Module):
    """每个 token 独立经过两层线性层；不同模态会拥有不同的 Mlp 实例。"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        # 初始化 PyTorch 模型基类。
        super().__init__()
        # 如果没有指定输出维度，就保持输入和输出维度相同。
        out_features = out_features or in_features
        # 隐藏层默认与输入同宽；VLMo 的 Block 通常显式设置更大的宽度。
        hidden_features = hidden_features or in_features
        # 第一层把每个 token 的特征从 D 扩到隐藏维度。
        self.fc1 = nn.Linear(in_features, hidden_features)
        # GELU 给前馈网络加入非线性表达能力。
        self.act = act_layer()
        # 第二层把隐藏向量映射回输出维度。
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 训练时随机丢弃部分激活，缓解过拟合。
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x 可是 [B,N,D]，线性层只改变最后一个特征维。
        x = self.fc1(x)
        # 先激活，再做 dropout。
        x = self.act(x)
        x = self.drop(x)
        # 回到 Block 所需的 D 维，并再做一次 dropout。
        x = self.fc2(x)
        x = self.drop(x)
        # FFN 本身不混合不同 token；跨 token 交流由 Attention 完成。
        return x


class Attention(nn.Module):
    """图像、文字、混合图文共用的多头自注意力。"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        # 登记注意力层的可训练参数。
        super().__init__()
        # H 个头会把总特征宽度 D 均分。
        self.num_heads = num_heads
        # 单个头的特征宽度为 D/H。
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # 点积除以 sqrt(单头维度)，防止维度大时 softmax 过于尖锐。
        self.scale = qk_scale or head_dim ** -0.5

        # 一次线性映射产生 Q、K、V 三份向量，权重形状 [3D,D]。
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # 若启用偏置，仅给 Q 和 V 保存可训练偏置；K 偏置用零补足。
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            # None 代表后续调用 F.linear 时没有偏置。
            self.q_bias = None
            self.v_bias = None
        
        # 注意力概率和输出投影后分别应用 dropout。
        self.attn_drop = nn.Dropout(attn_drop)
        # 把多个头拼接后的 D 维结果再做线性变换。
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        # B=batch 数，N=token 数，C=特征宽度。
        B, N, C = x.shape

        # 默认不使用 QKV 偏置。
        qkv_bias = None
        if self.q_bias is not None:
            # Q、零 K、V 依次拼成长度 3D 的偏置；K 中间的零不训练。
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # 用 [3D,D] 权重一次计算 QKV，输出 [B,N,3D]。
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # 拆成 [3,B,H,N,每头维度]，第一维分别代表 Q、K、V。
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q、k、v 各自为 [B,H,N,每头维度]。
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # 缩放 Q 与缩放最终 QK 点积等价。
        q = q * self.scale
        # Q 乘 K 的转置，得到 [B,H,N,N] 的 token 两两相关分数。
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            # 相对位置偏置告诉模型两个 token 在图片/文字中的相对距离。
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # 输入有效位置为 True；无效的 PAD 不能被其他 token 读取。
            mask = mask.bool()
            # 两个 None 维度让 [B,N] 的 mask 广播成 [B,1,1,N]。
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        # 按最后一维转为概率；type_as(x) 恢复原来的精度。
        attn = attn.softmax(dim=-1).type_as(x)
        # 训练时随机丢弃一部分注意力概率。
        attn = self.attn_drop(attn)

        # 概率加权 V，先得到 [B,H,N,每头维度]，再拼回 [B,N,C]。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 所有头拼接后的输出再经过一层投影和 dropout。
        x = self.proj(x)
        x = self.proj_drop(x)
        # 返回自注意力输出，外层 Block 会负责残差连接。
        return x


class Block(nn.Module):
    """一个 Transformer 块：共享 Attention，按模态选择 FFN 专家。"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
    ):
        # 让 PyTorch 登记层参数。
        super().__init__()
        # 注意力之前先做 LayerNorm，稳定输入数值。
        self.norm1 = norm_layer(dim)
        # 自注意力由所有模态共用；“专家”主要体现在后面的不同 FFN。
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 有概率跳过残差分支；概率为 0 时使用不改变输入的 Identity。
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 文字和图像专家各有独立 LayerNorm 参数。
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        # FFN 隐藏层通常是输入维度的 4 倍。
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 文字 token 专用的前馈网络。
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # 图像 patch token 专用的前馈网络。
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # 默认不创建图文专用 FFN；此时混合序列会拆开走两种专家。
        self.mlp_vl = None
        if with_vlffn:
            # 配置允许时，图文混合序列可使用第三套融合 FFN。
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            # 图文专家也有自己独立的归一化层。
            self.norm2_vl = norm_layer(dim)
        
        # gamma_1 和 gamma_2 是 LayerScale：分别调节注意力与 FFN 残差强度。
        # 若初始化值是 None，就退化为常数 1，不额外训练缩放参数。
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        # 无图文专用 FFN 时，前 max_text_len 个 token 被视为文字。
        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        # x 的形状为 [B,N,D]；先让所有 token 经同一套自注意力交流。
        # 残差连接：保留原 x，再加共享注意力产生的新信息。
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "image":
            # 只有图像输入，走图像 FFN。
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "text":
            # 只有文字输入，走文字 FFN。
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        else:
            # modality_type 为图文混合时，要么使用第三套 FFN，要么按模态拆开。
            if self.mlp_vl is None:
                # 混合输入且没有专用融合 FFN：按文字最大长度切成两段。
                x_text = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len :]
                # 注意力已经让两种 token 交流；这里只对文字 token 用文字 FFN。
                x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                # 图像 token 则使用参数独立的图像 FFN。
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                # 各自走完对应 FFN 后，按原 token 顺序拼回统一序列。
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                # 有图文专用 FFN 时，整条混合序列一起处理。
                x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))

        # 返回与输入同形状的 token 序列，交给下一层 Block。
        return x


class PatchEmbed(nn.Module):
    """把整张图像切为不重叠 patch，并将每块投影成一个向量。"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        # 初始化模型基类。
        super().__init__()
        # 224 变成 (224,224)，方便同时处理高和宽。
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 图片每个方向的块数相乘，得到 patch 总数。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # 保存二维 patch 网格大小，后续位置偏置需要知道行列。
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 这三个字段供检查输入尺寸和构建 CLS/位置向量。
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 核大小=步长=patch 大小，每块像素直接映射到 embed_dim 维。
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            # 配置可关闭卷积偏置；默认保留。
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        # B 张图片、C 个颜色通道、H 高、W 宽。
        B, C, H, W = x.shape
        # 原实现要求图片尺寸精确匹配构造模型时设定的 img_size。
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # FIXME look at relaxing size constraints
        # 输出仍是二维网格：[B,embed_dim,网格高,网格宽]。
        x = self.proj(x)
        return x


class MultiWayTransformer(nn.Module):
    """VLMo 的视觉骨干：图像 patch 嵌入和多层共享注意力／多路 FFN。

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        need_relative_position_embed=True,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        vlffn_start_layer_index=10,
        config=None,
    ):
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
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            vlffn_start_layer_index (int): vl-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        # 注册本模块的可训练参数。
        super().__init__()
        # 若传入全局配置，以其中的随机深度比例为准。
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        # 只在主进程打印随机深度比例。
        rank_zero_info("drop path rate: {}".format(drop_path_rate))
        # 绝对位置向量和相对位置偏置是两种不同的位置表达方式。
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        # 记录每个 token 的隐藏维度，外部 VLMo 模块还要读取。
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        # 没指定归一化层时，使用数值稳定的 LayerNorm(eps=1e-6)。
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # 图像经过 PatchEmbed 后从像素网格变成 patch 网格。
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        # CLS 之外的位置向量数量取决于图片共有多少 patch。
        num_patches = self.patch_embed.num_patches
        # 记录后面构建位置偏置和图文专家边界所需的信息。
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index
        # 纯文本 MLM 阶段不启用图文专用 FFN，把开始层移到网络末尾。
        if config["loss_names"]["textmlm"] > 0:
            self.vlffn_start_layer_index = depth
            rank_zero_info("Set vlffn_start_layer_index={} for text-only pretraining".format(self.vlffn_start_layer_index))
        # 图像 CLS 是可学习的全局汇总 token，形状 [1,1,D]。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 选择绝对位置嵌入时，每个 patch 和 CLS 各有一条可学习位置向量。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None
        # 在图像 token 进入 Transformer 前做一次 dropout。
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 第 i 层的 DropPath 比例从 0 逐渐增加到 drop_path_rate。
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # ModuleList 把 depth 个 Block 注册为模型的一部分。
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    # 后面若干层启用图文融合 FFN，前面层只有文字和图像专家。
                    with_vlffn=(i >= self.vlffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    # 混合序列前 max_text_len 个位置属于文字。
                    max_text_len=config["max_text_len"],
                )
                for i in range(depth)
            ]
        )
        # 最后一层 Block 后再做一次归一化。
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            # 截断正态分布初始化位置向量，避免初始值过大。
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # 对内部线性层和归一化层应用下方的初始化规则。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 线性层权重用小方差截断正态分布，偏置置零。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # LayerNorm 初始为 y≈x：偏置 0、缩放 1。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # JIT 跟踪时忽略这个供优化器查询的辅助函数。
    @torch.jit.ignore
    def no_weight_decay(self):
        # 位置嵌入与 CLS 不参加权重衰减。
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):
        # 输入图片形状 [B,3,H,W]，卷积后得到 [B,D,网格高,网格宽]。
        x = self.patch_embed(_x)
        # 展平网格并换轴，得到 [B,patch 数,D]。
        x = x.flatten(2).transpose(1, 2)
        # L 是 patch 数，B 是图片数量。
        B, L, _ = x.shape

        # 同一个 CLS 参数供 B 张图使用，expand 不新增独立参数。
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 把 CLS 放在第 0 个位置，后面依次是图像 patch。
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            # 绝对位置配置下给每个 token 加上自己的位置向量。
            x = x + self.pos_embed
        # 完成 token 化后施加 dropout。
        x = self.pos_drop(x)

        # 没有图像 PAD，每张图的所有位置都有效，用 1 表示。
        x_mask = torch.ones(x.shape[0], x.shape[1])

        # 同时返回 token 和有效位；上层 infer() 会把它与文字 mask 拼接。
        return x, x_mask


# VLMo base/p16
@register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    # 从可选参数取图片尺寸，没有提供则用 224×224。
    img_size = kwargs.pop("img_size", 224)
    # Base：12 层、宽度 768、12 个注意力头；第 10 层起使用图文 FFN。
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=10, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # timm.create_model 会收到这个实例。
    return model

# VLMo large/p16
@register_model
def vlmo_large_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    # Large：24 层、宽度 1024、16 个头；第 21 层起使用图文 FFN。
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo base+/p16
@register_model
def vlmo_base_plus_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    # Base+：24 层但宽度 544，使用绝对位置嵌入且不建相对位置偏置。
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=544, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21,
        use_abs_pos_emb=True, need_relative_position_embed=False, 
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
