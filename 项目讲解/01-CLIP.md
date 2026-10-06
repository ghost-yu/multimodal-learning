# 01｜CLIP：把图片和句子放到同一个向量空间

> **阅读入口**：建议配合 [论文与代码解释](../Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision/解释.md)，直接打开仓库里的 [模型](../CLIP/clip/model.py)、[加载与 tokenize](../CLIP/clip/clip.py)、[BPE 分词器](../CLIP/clip/simple_tokenizer.py)。这三个原文件已补中文行内注释，解释变量命名与来源、库函数、形状和原理。

> 适合第一篇读。源码：[CLIP](../CLIP)；论文：[CLIP.pdf](../papers/CLIP.pdf)。本篇解释这份官方仓库实际提供的**模型与推理代码**。它没有原论文的完整预训练循环。

## 一、先用一句话理解它

CLIP 收到一张图片和若干文字描述，分别把图片、文字编码成数字向量，比较哪个文字向量与图片向量最接近。比如图片是一只猫，文字候选是“猫”“狗”“汽车”，模型会为三个候选分别打分。

整条链是：

```text
图片 → 图片预处理 → 图像编码器 → 图像向量 ┐
                                         ├→ 向量相乘 → 图文相似度
句子 → 分词与整数 ID → 文本编码器 → 文本向量 ┘
```

`图像编码器` 和 `文本编码器` 是两套网络；图文在最后比较向量时才相遇。这叫**双编码器**。因为两边可以提前分别编码，检索大量图片或文字比较方便。

## 二、先认清五个文件

| 文件 | 在项目中的工作 | 第一次读哪部分 |
|---|---|---|
| [README.md](../CLIP/README.md) | 最短推理示例 | `model, preprocess = clip.load(...)` 开始 |
| [clip/clip.py](../CLIP/clip/clip.py) | 模型列表、权重下载、图片预处理、文本 token 化 | `load()`、`tokenize()` |
| [clip/model.py](../CLIP/clip/model.py) | 图像编码器、文本编码器、相似度计算 | `VisionTransformer`、`CLIP` |
| [clip/simple_tokenizer.py](../CLIP/clip/simple_tokenizer.py) | 把文字变成整数 ID | `SimpleTokenizer.encode()` |
| [tests/test_consistency.py](../CLIP/tests/test_consistency.py) | 检查模型不同加载方式的输出一致性 | 理解主链后再看 |

## 三、从 README 的使用代码逐步追进去

### 第 1 步：`clip.load()`

源码在 [clip/clip.py:94](../CLIP/clip/clip.py)。`clip.load("ViT-B/32", device)` 返回两个对象：

- `model`：已经建好结构并载入权重的 CLIP 模型。`ViT-B/32` 中 `B` 是 base 规格，`32` 是 ViT 切图的 patch 边长。
- `preprocess`：把 PIL 图片变成模型需要的张量。它负责调整大小、裁剪、转 RGB、转换为数字并归一化。它不是模型网络，但必须与训练时用的输入习惯一致。

`load()` 先判断你传的是内置模型名还是本地权重路径。内置名会触发下载；随后尝试按 JIT 文件读取，普通调用默认构建可继续查看的 PyTorch 模型。`build_model(state_dict)` 在 [model.py:399](../CLIP/clip/model.py) 根据权重中的参数形状推断模型结构，再把权重装回去。**只克隆源码不会自动下载模型权重**，第一次调用内置模型名才可能下载。

### 第 2 步：`clip.tokenize()`

源码在 [clip/clip.py:205](../CLIP/clip/clip.py)。输入是字符串列表，输出是整数矩阵 `[句子数,77]`。先由 BPE tokenizer 将文字拆成 token ID，再加开始符和结束符，空位补齐。所谓 token 可以是单词、词的一部分或标点；计算机此时只看到整数。

**容易误解**：`tokenize()` 还没有让模型理解语义，只是把输入变成模型能接收的格式。语义表示在后面的 embedding 和 Transformer 中形成。

## 四、图像编码器逐步看

打开 [VisionTransformer.forward():223](../CLIP/clip/model.py)。假设图片大小 `224×224`，patch 边长 32，batch 是 2。

| 代码阶段 | 做了什么 | 形状 |
|---|---|---|
| 输入图片 | RGB 图片 | `[2,3,224,224]` |
| `self.conv1(x)` | 每个 `32×32` 小块变成一个向量 | `[2,width,7,7]` |
| `reshape`、`permute` | 7×7 网格展开为序列 | `[2,49,width]` |
| 拼 `class_embedding` | 额外放一个汇总 token | `[2,50,width]` |
| 加 `positional_embedding` | 注入 patch 的位置信息 | `[2,50,width]` |
| Transformer | patch 互相注意 | `[2,50,width]` |
| 取 `x[:,0,:]` 并投影 | 得到每张图的整体向量 | `[2,embed_dim]` |

`conv1` 看起来像普通卷积，但在这里它的核大小与步长都等于 patch 大小，主要作用是**切成不重叠的块并做线性映射**。`class_embedding` 是一个可训练的特殊向量；它经过注意力后汇集其他 patch 的信息。`positional_embedding` 是必要的，因为展平后序列本身不能直接表明一个 patch 原来在左上还是右下。

如果 `vision_layers` 是列表，`CLIP.__init__()` 会走 [ModifiedResNet:94](../CLIP/clip/model.py) 分支；其尾部使用 [AttentionPool2d:58](../CLIP/clip/model.py) 汇聚特征。第一次读时先把 ViT 分支弄懂即可。

## 五、文本编码器逐步看

打开 [CLIP.encode_text():343](../CLIP/clip/model.py)。假设 3 句话，每句补到 77 个 token：

1. `self.token_embedding(text)` 把 `[3,77]` 的整数 ID 转成 `[3,77,transformer_width]` 的向量序列。embedding 是可训练的查找表。
2. 加位置编码后，转为 `[77,3,transformer_width]`，因为这份 Transformer 实现把序列长度放第一维。
3. 经过带因果 mask 的 Transformer。因果 mask 在 [build_attention_mask():328](../CLIP/clip/model.py)：一个位置不能读取它后面的词。因此结束符位置能汇总之前的整句。
4. 转回 batch 在前的形状，并按 `text.argmax(dim=-1)` 找结束符 EOT 的位置。这里能用 `argmax` 是由于这份 tokenizer 的 EOT ID 最大，**不是所有 tokenizer 都能这样找**。
5. 取 EOT 的向量后乘 `text_projection`，得到 `[3,embed_dim]`。现在图像向量与文本向量维度相同，才能直接比较。

## 六、相似度和训练信号

[CLIP.forward():358](../CLIP/clip/model.py) 调用 `encode_image` 和 `encode_text`，将两组向量各自除以 L2 长度，再算：

```text
logits_per_image = exp(logit_scale) × image_features × text_featuresᵀ
```

如果 2 张图、3 句话，形状是 `[2,embed_dim] × [embed_dim,3] → [2,3]`。第 `i,j` 格就是图 `i` 与句子 `j` 的分数。归一化后的点积对应余弦相似度；`exp(logit_scale)` 把分数拉到合适的训练尺度，且保证缩放为正。

原论文训练时，通常每个 batch 放 `B` 对配好的图文，形成 `[B,B]` 矩阵，对角线是正样本，其他格提供负样本。对每行和每列各算交叉熵再平均。**这份官方代码的 `forward()` 只返回两个方向的 logits，没有计算交叉熵，也没有训练脚本**。不要在源码里苦找不存在的 `train.py`。

## 七、第一次自己做什么实验

只用 README 的推理示例，取一张图和三个英文提示：`a photo of a cat`、`a photo of a dog`、`a photo of a car`。打印 `logits_per_image.shape` 与 `softmax` 概率；替换图片再观察分数。第二个实验把候选改成完整描述，观察短词和自然句子的差异。此实验使用预训练权重，不等于重做 CLIP 的海量数据训练。

**本项目完成标准**：你能说出 `tokenize()`、`encode_image()`、`encode_text()`、`forward()` 各自返回什么，并独立解释为何 2 张图、3 句话得到 `[2,3]` 的分数矩阵。
