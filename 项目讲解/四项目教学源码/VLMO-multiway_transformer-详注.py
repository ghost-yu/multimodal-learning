# 教学版：原文件 UniLM/vlmo/vlmo/modules/multiway_transformer.py。只增加注释。
# B=batch，N=token 总数，D/C=特征宽度，H=注意力头数，d=D/H。
# import：partial 预填构造参数；torch 张量；nn 层；F 函数式运算；timm 提供 DropPath/注册/初始化。
# rank_zero_info 只在 Lightning 主进程记日志。Block 的 attention 共享，FFN 按 modality_type 分路。
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
# functools 是 Python 标准库；partial(函数, 指定参数) 返回新可调用构造器。
# 例如 partial(nn.LayerNorm,eps=1e-6)(768) 等于 nn.LayerNorm(768,eps=1e-6)。
# partial 用来预先指定 LayerNorm 的 eps 等参数。
from functools import partial

# torch 是 PyTorch 张量主库：ones/zeros 建张量，cat 拼接，linspace 生成均匀数列。
# nn（neural network）定义 Module/Linear/Conv2d/LayerNorm/Parameter 等可训练结构。
# F 是 torch.nn.functional 的别名；F.linear 在已有权重上做函数式线性变换。
# torch 是张量计算主体，nn 定义网络层，F 提供函数式线性层。
import torch
import torch.nn as nn
import torch.nn.functional as F

# timm 是视觉模型库。DropPath 训练时随机跳过整条残差分支；to_2tuple(224)→(224,224)；
# trunc_normal_ 在已有张量上原地填截断正态随机数，末尾 _ 通常提示原地修改。
# DropPath 随机跳过残差分支；to_2tuple 把一个整数变成高宽两个整数。
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# register_model 是装饰器，让 timm.create_model 按函数名找到下面三个 VLMo 规格构造函数。
# 注册模型名，之后 VLMo 可用 timm.create_model("vlmo_base_patch16") 创建。
from timm.models.registry import register_model
# Lightning 的 rank_zero_info 只让主进程打印配置，避免多 GPU 每卡输出同一行。
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
        # in_features 是输入 D，hidden_features 是 FFN 中间宽度，out_features 是输出宽度；
        # act_layer 是激活函数的“类”（默认 GELU），drop 是训练时随机丢弃的概率。
        # 初始化 PyTorch 模型基类。
        super().__init__()
        # 如果没有指定输出维度，就保持输入和输出维度相同。
        # Python 的 or 取前一个真值，否则用输入宽度；输出默认与输入同宽才能做残差相加。
        out_features = out_features or in_features
        # 隐藏层默认与输入同宽；VLMo 的 Block 通常显式设置更大的宽度。
        hidden_features = hidden_features or in_features
        # 传 None 或其他假值时用 in_features；VLMo 的 Block 明确给 hidden_features=dim×mlp_ratio。
        # 第一层把每个 token 的特征从 D 扩到隐藏维度。
        self.fc1 = nn.Linear(in_features, hidden_features)
        # GELU 给前馈网络加入非线性表达能力。
        self.act = act_layer()
        # act_layer() 实例化激活层；不是立即对输入 x 做计算，计算发生在 forward。
        # 第二层把隐藏向量映射回输出维度。
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 训练时随机丢弃部分激活，缓解过拟合。
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x 在该函数内是当前 token 特征；每次 x=... 会覆盖局部绑定，形状按下一行更新。
        # x 可是 [B,N,D]，线性层只改变最后一个特征维。
        x = self.fc1(x)
        # 先激活，再做 dropout。
        x = self.act(x)
        x = self.drop(x)
        # 回到 Block 所需的 D 维，并再做一次 dropout。
        x = self.fc2(x)
        # 第二次 dropout 也仅训练模式生效；返回的形状与输入相同才能残差相加。
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
        # dim=D；num_heads=H；qkv_bias 控制 Q/V 可训练偏置；qk_scale 可覆盖默认 1/sqrt(d)。
        # attn_drop 丢注意力概率，proj_drop 丢注意力输出投影，两者位置不同。
        # 登记注意力层的可训练参数。
        super().__init__()
        # H 个头会把总特征宽度 D 均分。
        self.num_heads = num_heads
        # 单个头的特征宽度为 D/H。
        head_dim = dim // num_heads
        # // 是整数除法；模型配置应让 dim 能被 num_heads 整除，否则后面的 reshape 会失败。
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # 点积除以 sqrt(单头维度)，防止维度大时 softmax 过于尖锐。
        self.scale = qk_scale or head_dim ** -0.5
        # 若 qk_scale 是 None，就用 d**-0.5；这里 Python `or` 对显式 0 也会退回默认。

        # 一次线性映射产生 Q、K、V 三份向量，权重形状 [3D,D]。
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # 这里虽写 nn.Linear，但 forward 直接取 self.qkv.weight 用 F.linear，可动态拼 Q/K/V 偏置。
        # 若启用偏置，仅给 Q 和 V 保存可训练偏置；K 偏置用零补足。
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
            # K 的偏置不单独定义；前向时插入不训练的零向量。
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
        # x:[B,N,D]；mask 可是 [B,N] 的有效位置布尔/0-1 张量；
        # relative_position_bias 通常给每个头、每对位置一个偏置 [H,N,N]。
        # B=batch 数，N=token 数，C=特征宽度。
        # 三变量解包张量形状：B 是样本数，N 是 token 数，C 是每 token 的特征数。
        B, N, C = x.shape

        # 默认不使用 QKV 偏置。
        qkv_bias = None
        if self.q_bias is not None:
            # torch.zeros_like(self.v_bias) 复制形状/dtype/device 但值全零；requires_grad=False 关闭梯度。
            # Q、零 K、V 依次拼成长度 3D 的偏置；K 中间的零不训练。
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # 用 [3D,D] 权重一次计算 QKV，输出 [B,N,3D]。
        # F.linear 对最后一维作矩阵乘；一次输出 3C，顺序存 Q/K/V 三份特征。
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # 一个 token 的 3D 输出分三段：前 D 是 Q，中 D 是 K，后 D 是 V。
        # 拆成 [3,B,H,N,每头维度]，第一维分别代表 Q、K、V。
        # reshape 分为三份和 H 个头；-1 自动得每头宽度；permute 后 [3,B,H,N,d]。
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q、k、v 各自为 [B,H,N,每头维度]。
        q, k, v = (
            # q/k/v 分别是 query/key/value。每个都为 [B,H,N,d]；原 qkv[0/1/2] 是切片视图。
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # 缩放 Q 与缩放最终 QK 点积等价。
        q = q * self.scale
        # 缩放后的 q 仍 [B,H,N,d]；等价于将 QK^T 的分数除以 sqrt(d)。
        # Q 乘 K 的转置，得到 [B,H,N,N] 的 token 两两相关分数。
        # float 临时用 float32；K 的最后两维转置，QK^T 得 [B,H,N,N] 两两 token 分数。
        attn = (q.float() @ k.float().transpose(-2, -1))
        # 此刻 attn 是原始相关“分数”而非概率；第 3 维 query，第 4 维 key。
        
        if relative_position_bias is not None:
            # 相对位置偏置告诉模型两个 token 在图片/文字中的相对距离。
            attn = attn + relative_position_bias.unsqueeze(0)
            # unsqueeze(0) 从 [H,N,N] 变 [1,H,N,N]，广播复制到 B 个样本。

        if mask is not None:
            # 输入有效位置为 True；无效的 PAD 不能被其他 token 读取。
            mask = mask.bool()
            # bool() 把非零变 True；~ 在下一行把有效位取反成为“需屏蔽位”。
            # 两个 None 维度让 [B,N] 的 mask 广播成 [B,1,1,N]。
            # None 新增两个维度，mask 从 [B,N] 广播成 [B,1,1,N]；无效 key 填 -inf。
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        # 按最后一维转为概率；type_as(x) 恢复原来的精度。
        # 在每个 query 的所有 key 上归一化成概率，再转回输入精度。
        attn = attn.softmax(dim=-1).type_as(x)
        # softmax 对每个 query 所面对的 N 个 key 归一化，和为 1；type_as 回到输入精度。
        # 训练时随机丢弃一部分注意力概率。
        attn = self.attn_drop(attn)
        # 现在 attn 表示被 dropout 后的注意力权重；训练时各行和未必仍等于 1。

        # 概率加权 V，先得到 [B,H,N,每头维度]，再拼回 [B,N,C]。
        # 概率对 V 加权求和，再把 H 个头拼回 C 维；这里 x 已是注意力输出。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # attn:[B,H,N,N] 乘 v:[B,H,N,d]→[B,H,N,d]；再交换 H/N 并拼头为 C=D。
        # 所有头拼接后的输出再经过一层投影和 dropout。
        x = self.proj(x)
        x = self.proj_drop(x)
        # 返回的是更新后的 token 向量 [B,N,D]，注意力概率 attn 不作为返回值。
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
        # dim 是 token 宽度 D；with_vlffn 决定本层是否另外创建图文专用 FFN。
        # max_text_len 是混合输入中文字段固定占据的前 T 个槽位；不是图片 token 数。
        # 让 PyTorch 登记层参数。
        super().__init__()
        # 注意力之前先做 LayerNorm，稳定输入数值。
        self.norm1 = norm_layer(dim)
        # 自注意力由所有模态共用；“专家”主要体现在后面的不同 FFN。
        self.attn = Attention(
            # 这一个 Attention 对象对 image/text/vl 三条路径共享参数；专家分流发生在下面 MLP。
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
        # nn.Identity() 是原样返回输入的层，drop_path=0 时不产生随机跳层。
        # 文字和图像专家各有独立 LayerNorm 参数。
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        # FFN 隐藏层通常是输入维度的 4 倍。
        mlp_hidden_dim = int(dim * mlp_ratio)
        # int(...) 转 Python 整数，给 nn.Linear 的维度参数；例如 768×4=3072。
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
            # 同一层可有 text/image/vl 三套 MLP 与对应 LayerNorm，但 attention 仍只有一套。
        
        # gamma_1 和 gamma_2 是 LayerScale：分别调节注意力与 FFN 残差强度。
        # 若初始化值是 None，就退化为常数 1，不额外训练缩放参数。
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        # Python 条件表达式：若给初始化值，gamma_1 是可训练 [D] 向量；否则常数 1.0。
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        # 无图文专用 FFN 时，前 max_text_len 个 token 被视为文字。
        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        # modality_type 是字符串路由标记：'image'、'text' 或其他值（上层传 'vl'）。
        # 本层所有路径先经同一注意力，然后才决定 FFN 专家；输入输出都 [B,N,D]。
        # x 的形状为 [B,N,D]；先让所有 token 经同一套自注意力交流。
        # 残差连接：保留原 x，再加共享注意力产生的新信息。
        # 预归一化→共享注意力→LayerScale gamma_1→随机深度→与输入相加，三种模态都执行这一行。
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "image":
            # 只有图像输入，走图像 FFN。
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "text":
            # 只有文字输入，走文字 FFN。
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        else:
            # `else` 不仅是字面 'vl'，任何非 'image'/'text' 值也会进入；调用方应传预期标记。
            # modality_type 为图文混合时，要么使用第三套 FFN，要么按模态拆开。
            if self.mlp_vl is None:
                # 混合输入且没有专用融合 FFN：按文字最大长度切成两段。
                # 混合序列前固定 max_text_len 个是文字；此切分依赖上层拼接时保持该布局。
                x_text = x[:, : self.max_text_len]
                # 剩下位置是图像 token；之前共享注意力已让两个段相互读取信息。
                x_imag = x[:, self.max_text_len :]
                # x_text:[B,T,D]，x_imag:[B,N-T,D]；同一 x 已完成跨模态自注意力。
                # 注意力已经让两种 token 交流；这里只对文字 token 用文字 FFN。
                x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                # 图像 token 则使用参数独立的图像 FFN。
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                # 各自走完对应 FFN 后，按原 token 顺序拼回统一序列。
                x = torch.cat([x_text, x_imag], dim=1)
                # cat 后仍为 [B,N,D]，保持原文字在前、图片在后的 token 顺序。
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
        # img_size=预期图片高宽，patch_size=每块边长，in_chans=输入颜色通道数，embed_dim=D。
        # no_patch_embed_bias=True 才关闭卷积偏置；默认 False 时保留偏置。
        # 初始化模型基类。
        super().__init__()
        # 224 变成 (224,224)，方便同时处理高和宽。
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 即使用户传整数，后续都统一用 (高,宽) 两个整数；原局部变量被重新赋值。
        # 图片每个方向的块数相乘，得到 patch 总数。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # 宽方向块数×高方向块数。例如 224/16=14，最终 14×14=196 个 patch。
        # 保存二维 patch 网格大小，后续位置偏置需要知道行列。
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # patch_shape 保留网格的二维行列信息，num_patches 只保留总数，作用不同。
        # 这三个字段供检查输入尺寸和构建 CLS/位置向量。
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 核大小=步长=patch 大小，每块像素直接映射到 embed_dim 维。
        self.proj = nn.Conv2d(
            # Conv2d 先按 [B,C,H,W] 解释输入；kernel_size=stride=patch 尺寸，所以各块不重叠。
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            # 配置可关闭卷积偏置；默认保留。
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        # x 是一批原始图片像素张量；这里返回二维 patch 网格，还未 flatten 成 token 序列。
        # B 张图片、C 个颜色通道、H 高、W 宽。
        B, C, H, W = x.shape
        # 原实现要求图片尺寸精确匹配构造模型时设定的 img_size。
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # assert 失败会抛 AssertionError。改变输入尺寸需同步考虑位置向量和相对位置索引。
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
        # 参数族：img_size/patch_size/in_chans 描述图片；embed_dim/depth/num_heads/mlp_ratio 描述骨干；
        # drop_rate/attn_drop_rate/drop_path_rate 描述三处不同随机化；绝对/相对位置配置决定位置表达。
        # config 是上层总配置字典，此实现后面会直接索引 config['loss_names']、['max_text_len']。
        # 注册本模块的可训练参数。
        super().__init__()
        # 若传入全局配置，以其中的随机深度比例为准。
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        # 即使构造函数单独传了 drop_path_rate，只要 config 非 None，就以字典值为准。
        # 只在主进程打印随机深度比例。
        rank_zero_info("drop path rate: {}".format(drop_path_rate))
        # 绝对位置向量和相对位置偏置是两种不同的位置表达方式。
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        # 记录每个 token 的隐藏维度，外部 VLMo 模块还要读取。
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        # Python 链式赋值：self.num_features 与 self.embed_dim 都被设为同一个整数 D。
        # 没指定归一化层时，使用数值稳定的 LayerNorm(eps=1e-6)。
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # norm_layer 是以后可调用的“层构造器”；尚未创建某一层的权重。

        # 图像经过 PatchEmbed 后从像素网格变成 patch 网格。
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        # CLS 之外的位置向量数量取决于图片共有多少 patch。
        num_patches = self.patch_embed.num_patches
        # 这里取的是上面 PatchEmbed 保存的整数 patch 总数，供下面可选绝对位置表分配长度。
        # 记录后面构建位置偏置和图文专家边界所需的信息。
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index
        # 纯文本 MLM 阶段不启用图文专用 FFN，把开始层移到网络末尾。
        if config["loss_names"]["textmlm"] > 0:
            # textmlm=纯文本 Masked Language Modeling。起始层=depth 时，i<depth 的所有层都不建 vl FFN。
            self.vlffn_start_layer_index = depth
            rank_zero_info("Set vlffn_start_layer_index={} for text-only pretraining".format(self.vlffn_start_layer_index))
        # 图像 CLS 是可学习的全局汇总 token，形状 [1,1,D]。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Parameter 进入 optimizer；expand 到 B 张图时共享这一份可训练参数。
        # 选择绝对位置嵌入时，每个 patch 和 CLS 各有一条可学习位置向量。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None
        # 若用相对位置偏置而不使用绝对位置表，此属性是 None；visual_embed 会跳过加法。
        # 在图像 token 进入 Transformer 前做一次 dropout。
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 第 i 层的 DropPath 比例从 0 逐渐增加到 drop_path_rate。
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # dpr = drop path rates；linspace 生成 depth 个从 0 到最大值的浮点张量，item 取 Python 数。
        # ModuleList 把 depth 个 Block 注册为模型的一部分。
        self.blocks = nn.ModuleList(
            # ModuleList 正式登记每个子层；普通 list 不会被 Module.parameters/state_dict 自动遍历。
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
                    # i 从 0 开始；前面层不建图文 FFN，达到 start index 的层开始创建。
                    layer_scale_init_values=layer_scale_init_values,
                    # 混合序列前 max_text_len 个位置属于文字。
                    max_text_len=config["max_text_len"],
                )
                for i in range(depth)
                # 列表推导式每次新建一个 Block，层间参数不共享；只有同一层的模态路径共享其 Attention。
            ]
        )
        # 最后一层 Block 后再做一次归一化。
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            # 截断正态分布初始化位置向量，避免初始值过大。
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # std 是初始化标准差，trunc_normal_ 会修改已有 Parameter 的数值。
        # 对内部线性层和归一化层应用下方的初始化规则。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Module.apply 把每个子模块 m 传到此函数；只对 Linear/LayerNorm 命中的层改权重。
        # 线性层权重用小方差截断正态分布，偏置置零。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 有些 Linear 设 bias=False；先判不为 None 才能初始化。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 初始为 y≈x：偏置 0、缩放 1。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # JIT 跟踪时忽略这个供优化器查询的辅助函数。
    @torch.jit.ignore
    def no_weight_decay(self):
        # 装饰器告知 TorchScript 忽略此辅助函数；返回的是参数名字集合，不是实际张量。
        # 位置嵌入与 CLS 不参加权重衰减。
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):
        # _x 是输入图片 [B,3,H,W]；前导下划线只是作者命名，不表示 Python 特殊语法。
        # 输入图片形状 [B,3,H,W]，卷积后得到 [B,D,网格高,网格宽]。
        x = self.patch_embed(_x)
        # 展平网格并换轴，得到 [B,patch 数,D]。
        # flatten(2) 仅摊平图像网格两维，transpose 将 [B,D,P] 变 [B,P,D]。
        x = x.flatten(2).transpose(1, 2)
        # L 是 patch 数，B 是图片数量。
        B, L, _ = x.shape
        # L=patch 数，_ 接收 D 但不再使用；x 现在是 [B,L,D]。

        # 同一个 CLS 参数供 B 张图使用，expand 不新增独立参数。
        # expand 是让 B 张图共享同一个 CLS 参数的视图，-1 代表该维保持原长度。
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 把 CLS 放在第 0 个位置，后面依次是图像 patch。
        x = torch.cat((cls_tokens, x), dim=1)
        # dim=1 是序列维，得到 [B,L+1,D]；CLS 位于索引 0，patch 从索引 1 开始。
        
        if self.pos_embed is not None:
            # 绝对位置配置下给每个 token 加上自己的位置向量。
            x = x + self.pos_embed
        # 完成 token 化后施加 dropout。
        x = self.pos_drop(x)
        # x 已是视觉 token 序列，不再是原图片像素；下游共享 Transformer 接收它。

        # 没有图像 PAD，每张图的所有位置都有效，用 1 表示。
        # 图像没有 padding，所以 mask 全 1；上层 infer 会把它移到图像设备并拼文字 mask。
        x_mask = torch.ones(x.shape[0], x.shape[1])
        # 源码默认在 CPU 创建此 mask；上层 VLMo.infer 会把它转到图片设备。它是 0/1 有效位。

        # 同时返回 token 和有效位；上层 infer() 会把它与文字 mask 拼接。
        return x, x_mask


# VLMo base/p16
@register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    # **kwargs 收集额外关键字参数成字典；pop('img_size',224) 取出图片边长并从字典移除，
    # 使后面的 **kwargs 展开时不会重复传 img_size。pretrained 形参在这个工厂函数里未使用。
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
    # Large 工厂函数与 Base 流程相同，只改变层数/宽度/头数/图文 FFN 起始层。
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
    # Base+ 采用 24 层、544 宽、绝对位置向量；layer_scale_init_values=None 令 gamma 为常数 1。
    img_size = kwargs.pop("img_size", 224)
    # Base+：24 层但宽度 544，使用绝对位置嵌入且不建相对位置偏置。
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=544, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21,
        use_abs_pos_emb=True, need_relative_position_embed=False, 
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
