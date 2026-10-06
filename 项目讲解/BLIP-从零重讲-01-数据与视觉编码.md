# 第 1 章：一条数据怎样变成图像 token

对照源码：[数据集](../BLIP/data/pretrain_dataset.py)、[数据工厂](../BLIP/data/__init__.py)、[视觉编码器](../BLIP/models/vit.py)、[创建 ViT](../BLIP/models/blip.py)。本章用 `224×224` 图片和 ViT Base 举例。

## 1. `import` 到底拿来了什么

| 源码写法 | 初学者理解 | 本项目具体用处 |
| --- | --- | --- |
| `import json` | Python 标准库，读取 JSON 文本 | `json.load(f)` 把文件转为 Python 列表/字典 |
| `import os` | 路径、文件系统工具 | `os.path.join()` 拼目录与文件名；`os.path.isfile()` 检查本地权重 |
| `import glob` | 按通配符找路径 | `glob.glob('*.json')` 找所有标注分片 |
| `from PIL import Image, ImageFile` | Pillow 图片读写库 | `Image.open()` 打开图；`ImageFile.LOAD_TRUNCATED_IMAGES` 控制截断图片 |
| `from torch.utils.data import Dataset, DataLoader` | 数据集协议与按批迭代器 | `Dataset[i]` 给单条；`DataLoader` 拼成一批 |
| `from torchvision import transforms` | 常用图片处理工具包 | 随机裁剪、翻转、张量转换、标准化 |
| `import torch` | 张量、自动求导及设备操作 | `torch.cat` 拼 token，`torch.zeros` 建参数初值 |
| `import torch.nn as nn` | 神经网络层 | `nn.Linear`、`nn.ModuleList`、`nn.Parameter` |
| `import torch.nn.functional as F` | 不一定有独立参数的函数式运算 | 某些模型中用 `F.normalize` 等 |
| `from functools import partial` | 预先固定函数的部分参数 | `partial(nn.LayerNorm, eps=1e-6)` 生成统一配置的层构造器 |
| `from timm... import PatchEmbed, DropPath, trunc_normal_` | timm 的视觉模型组件 | 切块投影、随机跳过残差分支、截断正态初始化 |
| `checkpoint_wrapper` | fairscale 的激活检查点包装器 | 重算部分前向以节省训练显存 |

`vit.py` 还导入 `_cfg`、`register_model`、`named_apply`、`adapt_input_conv`，主要给代码后面的预训练权重载入、模型注册等逻辑使用；**读 BLIP 的前向过程时先不用追它们**。`pretrain_dataset.py` 中 `random` 和第二次 `os` 是原仓库保留导入，这段数据读取路径不使用它们。导入某个函数不等于本文件一定调用了它。

## 2. 数据集逐步读

### `class pretrain_dataset(Dataset)`

`class` 定义类；括号内的 `Dataset` 表示继承 PyTorch 数据集协议。至少要实现 `__len__`（多少条）和 `__getitem__`（第几条）。`self` 指当前这个数据集实例，不是特殊关键字，而是 Python 的惯用名字。

构造器 `__init__(self, ann_file, laion_path, transform)` 的三个名字：

- `ann_file`：**ann**otation file，标注文件路径列表。为什么是列表？训练配置可以合并多个 JSON。
- `laion_path`：LAION 数据分片所在目录；空值代表本次不用。
- `transform`：传进来的图片处理流水线，是可调用对象；`self.transform(image)` 就像调用函数。

`self.ann_pretrain = []` 先创建空列表。循环 `for f in ann_file` 中的 `f` 是**本轮标注文件路径**，不是图片，也不是文件对象。`json.load(open(f,'r'))` 将 JSON 读成 Python 对象，通常是若干 `{image:..., caption:...}` 字典组成的列表；`self.ann_pretrain += ann` 把当前文件的条目追加到总列表。这个写法没有用 `with open(...)` 管理文件，是原代码风格，读懂即可。

若 `laion_path` 非空，`glob.glob(os.path.join(laion_path,'*.json'))` 返回匹配路径列表，存为 `self.laion_files`。`self.laion_files[0]` 取第一份；若目录下没有 JSON，会发生 `IndexError`，原代码没有提前报更友好的错。读出的 `self.ann_laion` 与基础标注列表拼成 `self.annotation`。`reload_laion(epoch)` 在另一个训练轮次使用 `epoch % len(self.laion_files)` 选文件；`%` 是取余，文件轮流循环。**注意：只有配置提供 LAION 路径时才可调用它。**

`__len__()` 返回 `len(self.annotation)`。`__getitem__(index)` 中 `index` 是整数序号；`ann = self.annotation[index]` 得到一条标注字典。`Image.open(ann['image'])` 按字典里的路径开图，`.convert('RGB')` 强制变成红绿蓝三通道，避免灰度/透明通道导致模型通道数不对。随后 `self.transform(image)` 把 PIL 图片变为 `[3,H,W]` 浮点张量。`pre_caption(ann['caption'],30)` 清理文字并限制词数；此时仍是 Python 字符串，**还没转 token ID**。最终 `return image, caption` 返回一对。

### `create_dataset('pretrain', config)`

[`data/__init__.py`](../BLIP/data/__init__.py) 中的 `config` 是配置字典，`config['image_size']` 通过键取值。`transforms.Compose([...])` 按列表顺序调用每个图像处理器：

1. `RandomResizedCrop`：随机找图片中的一块并缩放到模型需要的正方形；`scale=(min_scale,1.0)` 控制裁剪面积占原图面积的范围。
2. `RandomHorizontalFlip`：随机左右翻转。
3. `RandomAugment`：从候选增强中抽操作调整外观。
4. `ToTensor()`：PIL 图片 `[H,W,3]` 语义转成 PyTorch `[3,H,W]`，典型像素值到 `[0,1]`。
5. `Normalize(mean,std)`：各通道做 `(值-均值)/标准差`，使输入与预训练视觉模型习惯的范围相符。

`transform_test` 采用固定 Resize、ToTensor、Normalize；没有随机裁剪，便于每次比较结果。`create_dataset` 的 `dataset` 参数是任务名字符串，分支进入 `pretrain_dataset(config['train_file'], config['laion_path'], transform_train)` 后又把局部变量 `dataset` 赋成数据集对象。**同名变量在这一函数内部由字符串变为对象**，是原代码的写法。

### `DataLoader` 与 batch

`create_loader()` 把数据集包成可迭代对象。`batch_size` 是一次取几条；`num_workers` 是并行读取进程数；`sampler` 决定索引顺序；`drop_last=True` 丢掉末尾不足一批的数据，便于队列和分布式训练的固定形状。若有 sampler，`shuffle=False`，因为顺序已经由 sampler 决定。假设 `batch_size=2`，两次 `__getitem__` 返回的图片堆成 `image: [2,3,224,224]`；文字合为两个字符串，供 tokenizer 下一步处理。

## 3. `create_vit` 的参数为什么这样命名

打开 [`models/blip.py`](../BLIP/models/blip.py) 的 `create_vit`：

| 名称 | 展开与意义 | 默认/例子 |
| --- | --- | --- |
| `vit` | Vision Transformer 的规格字符串 | `'base'` 或 `'large'` |
| `image_size` | 输入正方形图片的边长 | `224` |
| `use_grad_checkpointing` | 是否用梯度检查点节省显存 | `False` |
| `ckpt_layer` | 对最后多少个 Transformer block 用检查点 | `0` |
| `drop_path_rate` | 最深层随机跳过残差分支的概率 | 常见 `0`/`0.1` |
| `vision_width` | 每个图像 token 的特征维度 | Base `768`，Large `1024` |
| `visual_encoder` | 返回的 ViT 模型对象 | `VisionTransformer(...)` |

`create_vit` 返回**两个值** `(visual_encoder, vision_width)`；左侧 `self.visual_encoder, vision_width = ...` 是 Python 元组拆包。`vision_width` 是整数，不是又一个模型。`base` 用 `depth=12, num_heads=12`；`large` 用 `depth=24, num_heads=16`。`patch_size=16` 表示每块 `16×16` 像素。

## 4. ViT 初始化：名字与形状

源码 [`VisionTransformer.__init__`](../BLIP/models/vit.py)。`self.patch_embed = PatchEmbed(...)` 调用 timm 的组件；在这里它把 `[B,3,224,224]` 变为 `[B,196,768]`。`num_patches` 即 `196`。这类切块投影通常由步长等于 patch 大小的卷积完成；看懂输入输出即可理解 BLIP 后续调用。

`self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))`：`cls` 是 **classification** 的缩写，源自 ViT 的分类汇总 token。虽用于 BLIP 对齐，这个名字沿用 ViT；`nn.Parameter` 告诉 PyTorch “这是训练时可更新的值”。`self.pos_embed` 是 **position embedding**，形状 `[1,197,768]`，给第 0 个 CLS 和 196 个块各配一个位置向量。若没有位置向量，模型难以知道块在左上还是右下。

`dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]`：`dpr` 是 **drop path rates**，每层的丢路径概率列表；`torch.linspace(a,b,n)` 生成从 `a` 到 `b` 的 `n` 个数；`.item()` 把单元素张量变为 Python 数值。`nn.ModuleList([...])` 像列表一样存 12/24 个 `Block`，但还会让 PyTorch 正确登记每层参数，普通 Python 列表不能可靠替代。列表推导式的 `i` 是**第几层**，`i>=depth-ckpt_layer` 表示只对靠后的若干层启用检查点。`self.norm = LayerNorm(...)` 是最后一次归一化。

`trunc_normal_(tensor,std=.02)` 在原张量上用截断正态分布填初值；末尾 `_` 是 PyTorch 约定的**原地修改**标记。`self.apply(self._init_weights)` 递归访问子模块，按类型初始化 Linear 和 LayerNorm。`no_weight_decay()` 返回 `{'pos_embed','cls_token'}`，供外部优化器规则决定是否跳过权重衰减；返回集合本身不会自动修改优化器。

## 5. `VisionTransformer.forward()` 按行推

先令 `x=[2,3,224,224]`：

| 代码 | 这行做什么 | 输出形状/变量来源 |
| --- | --- | --- |
| `B = x.shape[0]` | 取第 0 维，一批图片数 | `B=2`。`x` 是调用 `visual_encoder(image)` 传入的批量图片 |
| `x = self.patch_embed(x)` | 切块并线性投影 | `[2,196,768]`；这里的 `x` 已不再是像素图 |
| `cls_tokens = self.cls_token.expand(B,-1,-1)` | 把一份 CLS 视图扩成每张图一份 | `[2,1,768]`；`-1` 表示该维保持原大小 |
| `x = torch.cat((cls_tokens,x),dim=1)` | 沿 token 维拼 CLS 和图像块 | `[2,197,768]`；`cat` 要求其他维一致 |
| `x = x + self.pos_embed[:,:x.size(1),:]` | 每个 token 加自己的位置向量 | 仍 `[2,197,768]`；`size(1)` 是 token 数 197 |
| `x = self.pos_drop(x)` | 训练时随机置零部分值 | 形状不变；评估模式通常不丢弃 |
| `for i,blk in enumerate(self.blocks)` | 依次取层号和层对象 | `i=0...11`，`blk` 是本层 `Block` |
| `x = blk(x, register_blk==i)` | 让本层处理全部 token | 形状仍 `[2,197,768]` |
| `x = self.norm(x)` | 特征归一化 | 仍 `[2,197,768]` |
| `return x` | 返回全部 token | 第 0 个是 CLS，后 196 个是局部块 |

`register_blk` 是为了可视化某一层注意力的层号；默认 `-1`，与实际层号不相等，所以通常不保存注意力图。

## 6. 一个 `Block` 到底算了什么

`Block.forward()` 只有两条主语句：

```python
x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
x = x + self.drop_path(self.mlp(self.norm2(x)))
```

读嵌套调用要从内向外：第一句先 `norm1(x)`，再 `attn(...)` 让不同 token 互相读取，再可能 `drop_path(...)`，最后加回旧 `x`。第二句先 `norm2(x)`，再 `mlp(...)` 对每个 token 的特征做非线性变换，然后加回上一句的 `x`。**这里两次出现的 `x` 是不断更新的局部变量。**“加回”称 residual connection（残差连接），要求两边形状相同。

`Mlp` 是 multi-layer perceptron，多层感知机；此处由 `fc1: 768→3072`、`GELU`、`Dropout`、`fc2: 3072→768` 组成。`fc` 是 fully connected（全连接）的惯用缩写。`nn.GELU` 提供非线性；连续两个纯线性层等价于一个线性变换，所以中间需要激活函数。`nn.Dropout(p)` 训练时随机置零元素；`DropPath` 随机跳过**整个残差支路**，二者粒度不同。

## 7. 注意力那一行 `reshape().permute()` 逐层拆开

[`Attention.forward()`](../BLIP/models/vit.py) 输入 `x:[B,N,C]`，其中 `N=197`、`C=768`、`num_heads=12`、每头 `head_dim=64`。

```python
qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
q,k,v = qkv[0],qkv[1],qkv[2]
attn = (q @ k.transpose(-2,-1)) * self.scale
attn = attn.softmax(dim=-1)
x = (attn @ v).transpose(1,2).reshape(B,N,C)
```

`qkv` 的英文是 Query/Key/Value，即“我想找什么/我有什么可匹配的/匹配后交给你什么信息”。`self.qkv` 是 `nn.Linear(C,3C)`，把一个 token 一次投影为三组向量，故 `[B,N,2304]`。`.reshape(B,N,3,12,64)` 只是重排元素的逻辑形状，得到 `[B,N,3,H,head_dim]`。`.permute(2,0,3,1,4)` 指定新维度依次取原第 2、0、3、1、4 维，于是 `[3,B,H,N,64]`。`qkv[0]` 取 Q，`qkv[1]` 取 K，`qkv[2]` 取 V，各 `[B,H,N,64]`。

`k.transpose(-2,-1)` 只交换最后两维，从 `[B,H,N,64]` 到 `[B,H,64,N]`。`@` 是矩阵乘法，每个 Query 与每个 Key 点积，结果 `[B,H,N,N]`。`self.scale = head_dim ** -0.5`，也就是 `1/sqrt(64)=1/8`，避免点积太大导致 softmax 几乎只选一个位置。`softmax(dim=-1)` 对“全部可读取 token”维度归一化，每行权重和为 1。`attn @ v` 把这些权重乘 V 并相加，仍是每个 Query 一个输出；之后把 12 个头合并回 `[B,N,768]`。最后 `self.proj` 是可训练的输出投影。

这里叫**自注意力**，因为 Q/K/V 都来自同一图像 token 序列。下一章的**交叉注意力**中，Q 来自文字，K/V 来自图像。

## 8. 三个常见误会

1. `image_embeds` 的 `embeds` 是 embeddings（向量表示），不表示模型已经生成图片。`[B,P,Dv]` 是图像特征。
2. `image_embeds[:,0,:]` 只取整体 CLS，适合 ITA；图文融合需要完整的 `[B,P,Dv]`，因为局部 patch 也要给文字读取。
3. `nn.Linear(768,256)` 不是把图片压成 256 个像素，而是把每个整体特征向量映射到 256 维共同空间。
