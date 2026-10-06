# CLIP 模型源码中文详注：视觉编码、文字编码、图文相似度和权重重建。
# import：OrderedDict 保持网络层命名顺序；Tuple/Union 只写类型提示；np 用于 log 初始化。
# torch 负责张量；nn 负责 Module/层/Parameter；F 是函数式算子（本文件在注意力池化使用）。
# OrderedDict 是标准库的有序映射：在 nn.Sequential 中给层取可读名字，且保持执行顺序。
from collections import OrderedDict
# Tuple/Union 只用于类型提示：vision_layers 可是四个整数的元组，也可是一个整数。
from typing import Tuple, Union

# numpy 的 log 是自然对数；这里只用于初始化可训练的 logit_scale，不处理图片数组。
import numpy as np
# torch 建/操作张量：randn 随机初始化，cat 拼接，arange 生成索引。
import torch
# F 是 torch.nn.functional 的别名。这里主要调用多头注意力的函数式实现。
import torch.nn.functional as F
# nn 包含 Module、Parameter、Conv2d、Linear、Embedding、LayerNorm 等网络构件。
from torch import nn


class Bottleneck(nn.Module):
    # 类继承 nn.Module，实例化后可放进 nn.Sequential 并被优化器遍历参数。
    # ResNet 的残差单元：主分支做三次卷积，旁路保留输入并在末尾相加。
    # expansion=4 表示输出通道数是 planes 的 4 倍；它是类属性，所有 Bottleneck 共用。
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        # 每造一个 Bottleneck 就执行本函数一次；三次卷积和可选旁路都在这里登记。
        # inplanes=输入通道，planes=中间瓶颈通道，stride=是否让空间高宽缩小。
        # super() 调 nn.Module 初始化，之后赋给 self 的层才会登记为子模块。
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        # BatchNorm2d 为每个输出通道保存缩放/偏移与运行均值方差；与纯数学标准化有可训练参数差异。
        self.bn1 = nn.BatchNorm2d(planes)
        # ReLU(x)=max(x,0)；inplace=True 尝试原地修改，减少中间内存。
        self.relu1 = nn.ReLU(inplace=True)

        # 3×3 卷积 padding=1 保持空间尺寸；第二层仍 stride=1，真正缩小由后面的 avgpool 做。
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第二次 BN 的通道数与 conv2 输出 planes 相等；ReLU 只改数值，不改形状。
        self.relu2 = nn.ReLU(inplace=True)

        # 需要降采样时先平均池化，避免直接跨步卷积造成较强混叠。
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 1×1 卷积只改变通道数，不混合邻近空间像素；这里从 planes 扩到 4×planes。
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # 最后 BN 的通道数是 4×planes；后面的 ReLU 在两分支相加之后执行。
        self.relu3 = nn.ReLU(inplace=True)

        # None 是“尚无旁路变换”；若高宽/通道不同，下面 if 会换成 Sequential。
        self.downsample = None
        # 把 stride 留作对象属性；forward 主要读已构建的 avgpool/downsample。
        self.stride = stride

        # 主分支与旁路的空间/通道形状必须相同，才能在 forward 末尾相加。
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                # 旁路也先平均池化，以便高宽与主分支匹配；名字 "-1" 只是层的字符串键。
                ("-1", nn.AvgPool2d(stride)),
                # 旁路 1×1 卷积把 inplanes 通道映射到 4×planes，与主分支末层同宽。
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        # x 是此残差块的输入特征图 [B,inplanes,H,W]；Tensor 类型提示不改变运行时。
        # identity 意为“恒等旁路”：保存进入这个 Bottleneck 的原始 x，最后与主分支相加。
        identity = x

        # 从内向外读：conv1(x)→bn1→relu1；out 是主分支当前结果，不是最终模型输出。
        out = self.relu1(self.bn1(self.conv1(x)))
        # 主分支第二次卷积继续处理 out；此处暂未与 identity 相加。
        out = self.relu2(self.bn2(self.conv2(out)))
        # 主分支需要 stride>1 时在此缩小高宽；Identity 情况下值与形状不变。
        out = self.avgpool(out)
        # 最后 1×1 卷积扩通道，并经 BatchNorm，输出形状与旁路必须一致。
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            # 只有旁路的维度与主分支不同时才需要 downsample；否则 identity 就是原输入。
            # 旁路也必须处理最初的 x，不能错误地拿已卷积的 out 处理。
            identity = self.downsample(x)

        # 残差连接：保留原输入信息，让深层网络更容易训练。
        out += identity
        # 相加后再激活；out += 是原地加法写法，形状两侧必须完全可加。
        out = self.relu3(out)
        # 返回本残差块输出，供同一 ResNet 阶段的下一个块接收。
        return out


class AttentionPool2d(nn.Module):
    # 与普通平均池化不同，这里只让一个汇总 query 读取全部空间位置的 key/value。
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        # 函数签名中的 spacial 是原作者拼法，含义是输入特征图每边的空间格点数。
        # spacial_dim（源码拼写）是特征图一边的格点数；embed_dim 是 ResNet 尾部通道数。
        # num_heads 把特征分为多个注意力头；output_dim 是最终图文共同向量维度。
        super().__init__()
        # +1 为前面将加入的全局汇总 token；除 sqrt(embed_dim) 控制随机初值尺度。
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # Q=query，K=key，V=value；三套线性投影各有独立可训练权重。
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        # q_proj/k_proj/v_proj 分别是不同的线性层；虽然输入输出宽度相同，权重不共享。
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # c_proj 把池化得到的内部向量映射到 CLIP 最终图文公共维度 output_dim。
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        # or 表示未传 output_dim 时保持 embed_dim；num_heads 只是整数配置，不是层。
        self.num_heads = num_heads

    def forward(self, x):
        # 这里 x 是 ResNet 最后一段输出的特征图 [B,embed_dim,spacial_dim,spacial_dim]。
        # ResNet 输出 [B,C,H,W]；展开成 [H*W,B,C] 的 token 序列。
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        # mean(dim=0) 在 H×W 个位置上取平均，得到每张图一个初始汇总向量 [1,B,C]。
        # 用所有空间位置的平均向量作为第 0 个汇总 token。
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # positional_embedding 原 [HW+1,C]；[:,None,:] 插入 batch 维，再广播到 B 张图。
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # 只有第 0 个 token 作为 query；所有位置提供 key/value，
        # 因此输出是注意力加权汇总的整张图向量，而非逐位置特征图。
        x, _ = F.multi_head_attention_forward(
            # query 只有第 0 个汇总 token，key/value 是包含所有位置的序列；故输出只有一个位置。
            query=x[:1], key=x, value=x,
            # 函数检查输入最后一维是否等于 embed_dim_to_check，帮助发现维度不匹配。
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            # 因分别传入 Q/K/V 权重，需把三套偏置拼成该函数期待的 [3C]。
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            # need_weights=False 不返回注意力权重矩阵，节省内存；左侧 _ 明确忽略第二返回值。
            training=self.training,
            need_weights=False
        )
        # 去掉长度为 1 的 query 维，得到 [B,output_dim]。
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    # CLIP 可用 ViT 或本类作视觉编码器，选择由 CLIP.__init__ 的 vision_layers 类型决定。
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        # layers 是每个残差阶段块数的四项元组；heads 是最后注意力池化的头数。
        # layers 是四段残差块数量，例如 (3,4,6,3)；width 是初始通道宽度，非图片像素宽。
        # input_resolution 是正方形输入边长；output_dim 是最终图像向量的维度。
        super().__init__()
        # output_dim 被最后 attention pool 使用；input_resolution 供图片预处理读取。
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        # stem 是输入图像进入四段残差层之前的连续卷积/池化模块。

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        # 第二/第三个 3×3 卷积继续提取局部图像特征，前两个用 width//2 通道。
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        # 第三个卷积把通道扩到 width，最后平均池化会把高宽各减半。
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        # _make_layer 会更新当前通道数，所以 _inplanes 是构建过程中的可变状态。
        self._inplanes = width  # this is a *mutable* variable used during construction
        # 这四次调用依次构造并登记层，_make_layer 内会更新 _inplanes 给下一段使用。
        self.layer1 = self._make_layer(width, layers[0])
        # layer2/3/4 的首块 stride=2，使空间高宽依次减半，通道逐段增加。
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        # layer3 的 planes=4×width；该段 Bottleneck 的真实输出通道为 4×planes。
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        # layer4 输出通道 width×8×expansion=width×32，传给注意力池化。
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # 这里 embed_dim 是 ResNet 尾部通道数，不是 CLIP 对比空间的 output_dim。
        embed_dim = width * 32  # the ResNet feature dimension
        # 四组残差层后不用普通全局平均池化，而用上面的注意力池化。
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        # 先建本段第一个 Bottleneck；它可能负责下采样或换通道。
        # planes=当前段瓶颈通道，blocks=本段 Bottleneck 数；首块可负责空间降采样。
        layers = [Bottleneck(self._inplanes, planes, stride)]

        # 首块输出通道为 4×planes；后续块必须按这个宽度接入。
        self._inplanes = planes * Bottleneck.expansion
        # range(1,blocks) 从第二个块开始，后续块都接收已扩展的通道数。
        for _ in range(1, blocks):
            # append 为列表增一个新 Bottleneck；这里 _ 是不需要使用的循环编号。
            layers.append(Bottleneck(self._inplanes, planes))

        # * 将每个 Block 展开为 Sequential 的位置参数；返回一个可调用阶段。
        return nn.Sequential(*layers)

    def forward(self, x):
        # 入口 x 是一批 RGB 图片 [B,3,H,W]；每阶段重新赋值给同名 x。
        def stem(x):
            # 内嵌函数只处理本次传入的图片张量 x；返回四段残差层之前的低级特征。
            x = self.relu1(self.bn1(self.conv1(x)))
            # stem 内的 x 已是第一层卷积特征，继续经第二层卷积/BN/ReLU。
            x = self.relu2(self.bn2(self.conv2(x)))
            # 第三层卷积输出 width 个通道。
            x = self.relu3(self.bn3(self.conv3(x)))
            # 平均池化把高宽各缩小 2 倍；stem 的输出供 layer1 使用。
            x = self.avgpool(x)
            # 此 return 只结束内嵌 stem()，不会结束外层 ModifiedResNet.forward()。
            return x

        # 把图片 float 精度改成首层卷积权重的精度，避免 float16/float32 混用。
        x = x.type(self.conv1.weight.dtype)
        # 输入数值精度与卷积权重一致，否则半精度模型可能报 dtype 不匹配。
        x = stem(x)
        # 每段把特征交给下一段；变量 x 一直重新绑定，最终才成为整图向量。
        x = self.layer1(x)
        # 后三段逐步缩小空间网格；每行的 x 是前一段输出，而非原始图片。
        x = self.layer2(x)
        # layer3 接 layer2 的输出，输出仍是 [B,C,H,W] 特征图。
        x = self.layer3(x)
        # layer4 结束后通道数为 width×32，空间尺寸约 input_resolution/32。
        x = self.layer4(x)
        # attnpool 将整张特征图汇总为 [B,output_dim]，此后 x 不再是四维特征图。
        x = self.attnpool(x)

        # 返回每张图一个向量，给 CLIP.forward 归一化并与文字向量点积。
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # 本类覆盖父类 forward：暂用 float32 计算归一化，再恢复入口半精度。
        # orig_type 保存入口张量原 dtype，如 float16；后面计算结束要转回它。
        orig_type = x.dtype
        # 半精度输入先用 float32 做归一化以提高数值稳定性，再转回原精度。
        ret = super().forward(x.type(torch.float32))
        # ret 是 float32 归一化结果，恢复 dtype 不改变形状。
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        # 逐元素近似 GELU：x * sigmoid(1.702*x)。输入输出形状完全相同。
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        # d_model 是每 token 宽度 D，n_head 是头数；attn_mask 为文本因果 mask，视觉分支一般为 None。
        super().__init__()

        # nn.MultiheadAttention 默认输入 [序列长度,B,D]，与后面 ViT 的 permute 对应。
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # ln_1/ln_2 各自保存独立缩放/偏移参数，一个用于注意力前，一个用于 MLP 前。
        self.ln_1 = LayerNorm(d_model)
        # OrderedDict 给层取名，initialize_parameters 后面可访问 block.mlp.c_fc。
        self.mlp = nn.Sequential(OrderedDict([
            # c_fc 把 D 扩到 4D，QuickGELU 非线性，c_proj 再压回 D 以便残差相加。
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # attn_mask 是文本注意力的可选加性矩阵；视觉块通常接 None。
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # x 是该残差块当前 token 序列 [长度,B,D]；视觉 mask=None，文本传因果 mask。
        # attn_mask 与当前张量放到相同设备/精度；文本分支的 mask 遮住未来词。
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 自注意力的 query/key/value 都来自 x；[0] 只取输出特征，不取权重。
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # 残差式预归一化：先保留旧 x，再加注意力的新信息；结果继续传给 MLP。
        # 先注意力再 MLP，每一步外层都有残差连接。
        x = x + self.attention(self.ln_1(x))
        # 此时 x 已加入注意力结果；第二条残差用 LayerNorm→前馈 MLP 逐 token 加工。
        x = x + self.mlp(self.ln_2(x))
        # 输入/输出形状相同，才能把本块直接接到下一个 Block。
        return x


class Transformer(nn.Module):
    # 多个同结构的 Block 顺序堆叠；各层是独立实例，不共享可训练参数。
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        # width=D，layers=堆叠层数，heads=每层多头数量；文本/图像可各建自己的实例。
        super().__init__()
        # 保存 D 和层数；CLIP.initialize_parameters() 后续读取它们计算初始化标准差。
        self.width = width
        self.layers = layers
        # * 将列表展开成 Sequential 的位置参数；Sequential 按层创建顺序依次调用。
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        # 列表推导式创建 layers 个独立 Block；* 展开为 nn.Sequential 的多个位置参数。

    def forward(self, x: torch.Tensor):
        # x:[长度,B,D]；所有层保持形状，改变的是 token 向量内容。
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    # ViT 将图片切成 patch 序列，经多层注意力后取 CLS 表示整张图。
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        # patch_size 是每个正方形小块边长；width 是 ViT 内部 D；output_dim 是最终与文字共享的维度。
        super().__init__()
        # input_resolution 供 clip.load 创建对应预处理；output_dim 是最终图文共享空间宽度。
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # 卷积核与步长同为 patch_size：[B,3,H,W]→[B,width,H/p,W/p]。
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # width**-0.5 即 1/sqrt(width)，避免后面可训练 CLS/位置向量初值过大。
        scale = width ** -0.5
        # class_embedding 只有一份可训练向量；forward 时广播复制到每张图的序列最前面。
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # // 是整数除法，求每边 patch 个数；平方是网格 patch 总数；+1 留给 CLS。nn.Parameter 使其被优化器训练。
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        # 对加完 CLS 与位置向量的每个 token 做归一化。
        self.ln_pre = LayerNorm(width)

        # 此 Transformer 只处理图像 token；CLIP 类内还有另一个处理文字的 self.transformer。
        self.transformer = Transformer(width, layers, heads)

        # 末层只归一化 CLS；[width,output_dim] 参数矩阵把视觉向量投到文字可比较空间。
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # 此函数的 x 是图片，跟 CLIP.encode_text() 里的文字 x 是不同函数的局部变量。
        # 输入 x: [B, 3, H, W]。卷积核大小和步长都等于 patch_size，
        # 所以每个不重叠的图像小块会被映射成一个 width 维向量。
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # 把二维 patch 网格展平成一维序列；-1 让 PyTorch 自动算 patch 个数。
        # reshape 的 -1 请 PyTorch 自动算 7×7 等 patch 总数；这里不改数值，只把 H/W 网格摊平。
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # Transformer 把 token 放在第 1 维，因此从 [B,width,P] 改成 [B,P,width]。
        # permute 的三个数字是原来的维度索引，不是目标长度：原 [B,D,P] → [B,P,D]。
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 在每张图的 patch 序列前加一个可训练的 CLS token，用于汇总整图信息。
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # zeros 的形状 [B,1,D] 用于把同一个 CLS 参数广播成 B 行；dtype/device 对齐当前 x。
        # 展平后的序列丢失了原来的二维位置；位置向量补回“哪个 patch 在哪里”。
        x = x + self.positional_embedding.to(x.dtype)
        # 当前 x:[B,P+1,width]，归一化不改变三维形状。
        x = self.ln_pre(x)

        # 注意 nn.MultiheadAttention 默认序列在第 0 维；把 [B,P+1,D] 换成 [P+1,B,D]。
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 在各 patch 与 CLS 间做多层自注意力，形状仍 [P+1,B,width]。
        x = self.transformer(x)
        # 换回 [B,P+1,width]，后面才能用 x[:,0,:] 逐图片取 CLS。
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 只取序列第 0 个 CLS token；它经过注意力后已汇集其他 patch 的信息。
        # x[:,0,:] 的三个索引依次是所有样本、第 0 个 CLS、全部特征；后面的 x 变成 [B,D]。
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            # 原类容许关闭投影；CLIP 构造时通常提供此可训练 Parameter。
            # 把视觉内部的 width 维映射到图文共用的 output_dim 维。
            x = x @ self.proj

        # 最终每张图片返回一个 [output_dim] 向量，不再返回全部 patch。
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        # embed_dim 是图文对比空间的共同宽度；image_resolution 是输入图片边长。
        # vision_layers 若为四项元组是 ResNet 各阶段块数，若为整数是 ViT 块数。
        # vision_width 是视觉内部通道宽；vision_patch_size 仅 ViT 使用。
        # context_length 是文本固定槽位数，vocab_size 是 token 种类数。
        # transformer_* 是文字 Transformer 的宽度、头数、层数，与视觉的参数分开。
        super().__init__()

        # 文本位置表按固定槽位数分配；应与 clip.tokenize() 使用的长度一致。
        self.context_length = context_length

        # vision_layers 为列表/元组时用 ResNet；为整数时用 ViT。
        # isinstance 做运行时类型判断：若层数写成四段列表/元组，构造 ResNet；整数构造 ViT。
        if isinstance(vision_layers, (tuple, list)):
            # ResNet 最终通道是 vision_width 的 32 倍；每头按约 64 维估算头数。
            vision_heads = vision_width * 32 // 64
            # 此分支的 self.visual 是 ResNet，forward 返回 [图数,embed_dim]。
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            # ViT 直接以 vision_width 作为注意力特征宽度，按每头约 64 维估算。
            vision_heads = vision_width // 64
            # 此分支的 self.visual 是 ViT，patch_size 参数在这里才有实际作用。
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 这里是文本 Transformer；视觉 Transformer 已经存放在 self.visual 中。
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        # 保存词表规模供权重恢复；真正查表由下面 token_embedding 执行。
        self.vocab_size = vocab_size
        # Embedding 相当于 [词表大小,文字宽度] 的可训练查找表；输入必须是词的整数 ID。
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # empty 仅分配 [T,D] 内存；initialize_parameters 随后为它填有效初始数值。
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # torch.empty 仅分配内存，内容未初始化；initialize_parameters() 会填正态随机值。
        self.ln_final = LayerNorm(transformer_width)

        # 文字内部 D_text 与共同 embed_dim 可能不同；矩阵乘把句向量映射过去。
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # 训练时比较的是“相似度 × exp(logit_scale)”；exp 保证缩放系数为正。
        # 存对数尺度而非尺度本身：forward 用 exp 还原正数；[] 代表标量张量。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 所有层建好后初始化，不是训练一步；加载 checkpoint 时权重还会被覆盖。
        self.initialize_parameters()

    def initialize_parameters(self):
        # normal_ 是原地写随机初值；这里不需要输入图片/文字 batch。
        # nn.init.normal_ 在已有 Parameter 上原地填 N(0,std²)；这几行不是训练步骤。
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # positional_embedding 的 empty 内存必须在使用前写入初始值。
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            # 只有实际建了 ResNet 分支，才访问 ResNet 专有的 attnpool 与四组 layer。
            if self.visual.attnpool is not None:
                # c_proj.in_features 是注意力池化输入宽度；-0.5 次方相当于除平方根。
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                # Q/K/V 三个投影及最终 c_proj 都用由输入宽度决定的标准差初始化。
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                # resnet_block 这里是整个阶段 Sequential，不是单个 Bottleneck。
                # named_parameters() 逐项返回 (参数名,Parameter)；name 是字符串，param 是张量参数。
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        # 将残差块末尾 BN 缩放设 0，使初始主分支较弱，更容易从恒等映射开始训练。
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # proj_std 给输出投影，深层网络用更小初始方差；下面两个 std 对应其他层。
        # 不同线性层用不同标准差；width 是文字 D，layers 是文字块数。
        attn_std = self.transformer.width ** -0.5
        # c_fc 先把文字 token 的 D 个特征扩展到 4D。
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            # block 是文本 Transformer 中当前 ResidualAttentionBlock；逐层初始化其中权重。
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            # out_proj 是多头结果合并后的线性映射；不是输出分类头。
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            # c_fc 扩维，c_proj 缩回 D；两个权重标准差不同。
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            # 文字投影参数 [transformer_width,embed_dim] 也要显式初始化。
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        # 先把所有词对的加性注意力偏置设为 -inf，再让 triu_ 清掉允许看的区域。
        mask.fill_(float("-inf"))
        # 上三角（未来 token）仍为 -inf，下三角和对角线变为 0；
        # 因此当前位置不能偷看后面的词。
        # triu_(1) 原地保留主对角线上方的 -inf；下方变 0，未来词因此被注意力屏蔽。
        mask.triu_(1)  # zero out the lower diagonal
        # 返回 [T,T] 的因果 mask，供所有文字残差注意力块复用。
        return mask

    @property
    def dtype(self):
        # @property 让 self.dtype 像普通属性一样读取；从视觉首层权重判断模型计算精度。
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        # encode_image 是薄包装：先把图片数值精度改到视觉卷积权重的 dtype，再调用视觉编码器。
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # text 是整数 ID，形状 [B,T]；查 embedding 表后变成 [B,T,D]。
        # 注意 text 在本函数是 tokenize() 的整数结果，不是 Python 字符串。
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # 位置矩阵 [T,D] 广播到每一句 [B,T,D]；告诉模型同一词在不同槽位的区别。
        x = x + self.positional_embedding.type(self.dtype)
        # 这份 Transformer 实现要求 [T,B,D]，因此交换 batch 与序列维。
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 文字序列现在按旧版 MultiheadAttention 的 [T,B,D] 顺序排列。
        x = self.transformer(x)
        # 经文字因果注意力后，x 中每个位置包含其可见的前文信息；形状仍 [T,B,D]。
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 每个文字位置最后 D 个特征归一化，然后转回视觉权重的 dtype。
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 每句话取 EOT 结束符位置的向量；本 tokenizer 的 EOT ID 最大，
        # 所以 argmax 能定位它。换 tokenizer 时这条假设未必成立。
        # arange(B) 给每句话行号；argmax 给各行 EOT 位置（仅因本 tokenizer 的 EOT ID 最大）；配对索引取 [B,D]。
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # 这次重新赋值后 x 不再是 token 序列，而是每句一个 [embed_dim] 向量。

        return x

    def forward(self, image, text):
        # image 是 [图片数,3,H,W] 浮点张量，text 是 [句子数,T] 整数张量；数量可不同。
        # 两个编码器分别输出整图/整句向量，均为 [B,D]（推理时两侧 B 可不同）。
        image_features = self.encode_image(image)
        # text_features 是文本 EOT 投影后的 [文字数,embed_dim]，此刻还未单位化。
        text_features = self.encode_text(text)

        # normalized features
        # L2 归一化后，两个向量点积就对应余弦相似度。
        # norm(dim=1) 算每张图的向量长度；keepdim 保持 [B,1]，按行广播相除后每行长度为 1。
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features 是 encode_text 的输出 [句子数,D]；同样 L2 归一化，点积才能解释为余弦相似度。
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # self.logit_scale 是训练参数（对数）；exp 后的 logit_scale 是本次前向里的正数张量。
        # [图片数,D] @ [D,文字数] -> [图片数,文字数]，每格是一个图文配对分数。
        # @ 为矩阵乘：[图数,D] × [D,句数] → [图数,句数]；每一行是该图面对所有文字候选的分数。
        logits_per_image = logit_scale * image_features @ text_features.t()
        # 转置后可用同一批分数计算“文字找图片”方向的训练目标。
        logits_per_text = logits_per_image.t()

        # 训练配对 batch 常使图/文数量相等；推理时允许矩形 [图片数,句子数]，不总是方阵。
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    # 函数名 convert_weights 指“改变权重张量的数值精度”，不改变模型层连接方式。
    # 这是把部分权重转换为 float16 的工具函数。fp16 更省显存；CPU 路径的 load() 随后转回 float32。
    # model 是整个 CLIP 实例；内部函数 l 是 Module.apply 递归访问到的某一层。

    def _convert_weights_to_fp16(l):
        # l 是 apply 正在访问的某一层；本内部函数不直接返回层，而是原地修改它的参数。
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # isinstance 只命中卷积和全连接层；Embedding 等类型不会走这段。
            # 参数张量的 .data.half() 改底层数值与 dtype；这是原实现的低层写法。
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                # bias=False 的层其 l.bias 是 None，此时不能调用 .data.half()。
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            # 该类有自己保存 Q/K/V 的参数，未必表现为普通 Linear 子层，所以单独转换。
            # 兼容合并 QKV 权重和分离 Q/K/V 权重两种命名；前导 * 展开列表。
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                # getattr(l,attr) 按字符串读取该层属性；可能为 None，故先判断。
                tensor = getattr(l, attr)
                if tensor is not None:
                    # 该版本/配置若没有对应参数，getattr 可能返回 None；只转换存在的张量。
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            # name 是属性名字字符串，下一行 hasattr 先确认当前 l 是否有这一属性。
            # 某些投影是直接登记在模型上的 Parameter，不属于 Linear，所以单独处理。
            if hasattr(l, name):
                # attr 此处是取出的 Parameter，不是参数名；它可能仍为 None。
                attr = getattr(l, name)
                if attr is not None:
                    # text_projection（CLIP）或 proj（ViT）是直接存放的矩阵参数。
                    attr.data = attr.data.half()

    # apply 会递归将内部函数应用于 model 的所有子模块；自身返回 model，但此处不接返回值。
    model.apply(_convert_weights_to_fp16)
    # apply 遍历所有子模块并执行内部转换函数，不返回新的模型实例。


def build_model(state_dict: dict):
    # state_dict 是字符串参数名→张量；本函数由权重形状反推出结构，再建 CLIP 并加载权重。
    # 检查权重名字中是否存在 ViT 特有的视觉投影参数，判断视觉骨干类型。
    vit = "visual.proj" in state_dict

    # vit 是布尔值；只由权重键是否含 visual.proj 推断视觉骨干类型。
    if vit:
        # conv1.weight 是 [视觉宽度,3,patch_size,patch_size]，第 0 维直接给 vision_width。
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        # 列表推导式只保留视觉注意力权重名，每个权重对应一个 ViT 残差注意力层。
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        # 返回的整数 vision_layers 将作为 ViT 构造参数，而非 ResNet 的四项元组。
        # 卷积核最后一维就是 patch 正方形边长。
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        # 位置向量总数 = grid_size² + 1，额外的 1 是 CLS token。
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        # 例如位置表长度 50：减 CLS 得 49，开平方得每边 7 个 patch；图片边长=7×patch_size。
        image_resolution = vision_patch_size * grid_size
    else:
        # ResNet 按 visual.layer1~4 分组，取每组不同块编号的数量；set 去重、len 计数。
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        # tuple(counts) 恢复各段 Bottleneck 数，如 (3,4,6,3)。
        vision_layers = tuple(counts)
        # layer1.0.conv1.weight 中第 0 维是该卷积输出通道，恢复 ResNet 的起始 width。
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        # 注意力池化位置表也有 1 个汇总 token，其余是最后特征图的方形网格。
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        # ResNet 注意力池化的位置数=输出网格边长²+1；先减 CLS 再开方并四舍五入。
        vision_patch_size = None
        # 防止位置表长度不能表示完整方形网格；失败说明权重结构与推断不符。
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        # ResNet 总共将边长下采样 32 倍，所以输出格点边长×32 恢复输入边长。
        image_resolution = output_width * 32
        # 四段 ResNet 总空间下采样系数是 32，恢复模型原输入图片边长。

    # 从检查点参数的形状反推出模型超参数，再新建同结构模型装权重。
    # text_projection 是 [文字内部宽度,共同空间宽度]，第二维给图文共同 embed_dim。
    embed_dim = state_dict["text_projection"].shape[1]
    # 以下逐个从参数张量形状恢复超参数；不能随意改与 checkpoint 不一致的维度。
    context_length = state_dict["positional_embedding"].shape[0]
    # token_embedding.weight 有 vocab_size 行，每行对应一个 token 的可训练向量。
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    # ln_final.weight 每个文字隐藏特征有一个缩放参数，长度即 transformer_width。
    transformer_width = state_dict["ln_final.weight"].shape[0]
    # 原实现按每头约 64 维估算头数；`//` 取整，需与原检查点结构相容。
    transformer_heads = transformer_width // 64
    # 对所有 transformer.resblocks.<层号>.* 的名字取层号并去重，得到文字层数。
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    # 权重键 transformer.resblocks.层号.参数名：split('.')[2] 取层号，set 去重后计层数。

    # 将上面推断的 10 个超参数按 CLIP.__init__ 的位置顺序传入；返回尚未装权重的新模型。
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        # 这三个 key 是旧权重文件可能额外保存的配置值，当前 CLIP 不登记同名 Parameter。
        # 老检查点可能把三个配置标量也存进 state_dict；它们不是当前模型的可训练参数。
        if key in state_dict:
            del state_dict[key]

    # 先将适用权重转半精度，再严格装载；.eval() 返回推理模式模型。
    convert_weights(model)
    # 默认 strict=True；权重键和模型参数必须匹配，否则抛错以避免静默漏装。
    model.load_state_dict(state_dict)
    # eval() 切换到评估模式并返回模型本身；不产生预测，推理仍需喂图片/文字。
    return model.eval()
