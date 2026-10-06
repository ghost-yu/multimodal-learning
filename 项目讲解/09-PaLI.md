# 09｜PaLI 系列：读论文，借 PaliGemma 代码理解图像到文本生成

> 本地论文：[PaLI](../papers/PaLI.pdf)、[PaLI-X](../papers/PaLI-X.pdf)、[PaLI-3](../papers/PaLI-3.pdf)。本地 [PaliGemma 参考代码](../big_vision-PaliGemma) 是**另一个模型**，不能称为 PaLI 原版源码或复现。这里明确区分“论文要学什么”和“本地代码实际展示什么”。

## 一、先说清项目关系

PaLI 系列研究视觉编码器、语言模型、多任务和多语言规模化。PaLI-X、PaLI-3 又分别研究更大的模型与改进视觉语言组件。规模很大，个人很难从零复制训练。PaliGemma 是受 PaLI-3 启发的开放模型，它公开了便于学习的图像前缀＋文字生成代码，但结构、权重、数据与 PaLI 系列并不完全一样。

因此本篇代码讲解的目标是：**看懂一个公开的图像加文字提示、逐词生成答案的模型**。论文的原始实验和各版本差异请直接看 PDF，勿从 PaliGemma 代码反推成“PaLI 一定这样实现”。

## 二、先把 PyTorch 和 JAX/Flax 语法对应起来

前八组大多是 PyTorch；这里的 Big Vision 代码使用 JAX/Flax：

| 这里的写法 | 可以怎么理解 |
|---|---|
| `jnp` | JAX 里的数组运算，概念上近似 `torch` |
| `jnp.concatenate([a,b],axis=1)` | 类似 `torch.cat([a,b],dim=1)` |
| `class Model(nn.Module)` | Flax 的模块定义，作用类似 PyTorch 模型类，但参数管理方式不同 |
| `__call__()` | 模块调用时执行的前向运算，在作用上接近 PyTorch `forward()` |
| `train=False` | 控制训练/推理时的行为，例如 dropout |

不要直接把 JAX 代码复制进 PyTorch 脚本；先理解张量形状和注意力规则。

## 三、只看这些文件

| 文件 | 工作 |
|---|---|
| [PaliGemma README](../big_vision-PaliGemma/big_vision/configs/proj/paligemma/README.md) | 模型、权重、任务使用说明 |
| [paligemma.py](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) | 图像/文本 embedding、拼接、注意力 mask、生成缓存 |
| [gemma_bv.py](../big_vision-PaliGemma/big_vision/models/proj/paligemma/gemma_bv.py) | 文字模型包装、token embedding、logits、缓存 |
| [vit.py](../big_vision-PaliGemma/big_vision/models/vit.py) | 图像 Transformer |
| [cococap.py](../big_vision-PaliGemma/big_vision/configs/proj/paligemma/transfers/cococap.py) | COCO caption 任务配置示例 |
| [transfers](../big_vision-PaliGemma/big_vision/configs/proj/paligemma/transfers) | 其他任务的微调配置 |

## 四、一张图和一段文字的前向路径

[Model.setup():61](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) 创建两个部件：`_img_model` 是图像编码器，`_llm` 是语言模型。它们首先各自处理输入：

1. `embed_image(image)` 把 `[B,H,W,3]` 图片变成图像 token `zimg=[B,P,D]`。如果输入视频帧，代码先把帧并到 batch 做视觉编码，再把多帧 token 拼回同一条序列。
2. `embed_text(tokens)` 把 `[B,T]` 文字 ID 查 embedding，得到 `ztxt=[B,T,D]`。
3. [embed_image_and_text():95](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) 使用 `jnp.concatenate([zimg,ztxt],axis=1)`，输出 `[B,P+T,D]`，图像 token 放在文字前面。这种“图像作为前缀”的序列让语言模型后续文字可以参考图像信息。
4. [__call__():131](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) 生成注意力 mask，调用 `_llm`，最后只取文字位置的 logits `[B,T,V]`。`V` 是词表大小，logits 是每个位置对下一词候选的未归一化分数。

注意这里没有将图像 patch 当成要输出的词。图像提供条件，训练目标在文字位置。

## 五、`make_attn_mask()` 为什么重要

[make_attn_mask():27](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) 用 `input_mask` 排除 padding，用 `mask_ar` 指定自回归边界。想象序列为 `[图像token们, 提示词, 答案词1, 答案词2]`：图像和提示可以作为前缀；生成答案词 1 时不能偷看答案词 2，否则训练分数很好看，但推理时没有未来答案可看。

源码对 `mask_ar` 做累积和，再比较两个位置的值，形成 `[B,N,N]` 布尔矩阵，其中 `N=P+T`。矩阵的第 `i,j` 格表示第 `i` 个 token 能否读取第 `j` 个 token。初读不必立即推导每个布尔运算，但必须知道它的目的：**保证预测时信息访问规则与生成时一致**。

## 六、生成为什么要缓存

生成时模型先看图像和已有提示，预测第一个答案词；把它接到后面，再预测下一个。若每步重新计算整张图与全部已生成词，计算重复很多。[prefill_cache():163](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) 先处理图像前缀和提示，[extend_cache():172](../big_vision-PaliGemma/big_vision/models/proj/paligemma/paligemma.py) 只把新词接进缓存。底层语言模型的实现可在 [gemma_bv.py](../big_vision-PaliGemma/big_vision/models/proj/paligemma/gemma_bv.py) 继续追踪。

## 七、初学者怎样学 PaLI 三篇

- **PaLI**：抓住“图像＋文本提示 → 文本输出”的统一任务接口，以及多语言训练的动机。
- **PaLI-X**：观察放大模型和扩大任务范围带来的工程变化，不要求复现训练配置。
- **PaLI-3**：重点看视觉编码器与系统整体设计如何调整，再和 PaliGemma README 描述的开放模型比较，分清论文模型与可用代码。

**本项目完成标准**：能解释 PaliGemma 的 `zimg`、`ztxt`、`mask_ar` 和 `text_logits`；能明确说出“这不是 PaLI 原版代码”。
