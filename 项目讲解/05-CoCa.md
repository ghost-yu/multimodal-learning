# 05｜CoCa：对比学习和生成学习合在一个网络里

完整阅读入口：[CoCa 论文与代码解释](../open_clip-CoCa/解释.md)。其中说明论文公式、本地版本差异、变量与 API、实验和复现状态。

> 论文：[CoCa.pdf](../papers/CoCa.pdf)。本地代码是 [OpenCLIP 的 CoCa 实现](../open_clip-CoCa)，不是 Google 原论文内部仓库。OpenCLIP 还包含许多其他模型，第一次只追 CoCa 相关文件。

## 一、先看它跟 CLIP 和 BLIP 的关系

CLIP 主要学“图和文的全局向量相不相似”；BLIP 也训练生成 caption。CoCa 用一个视觉编码器、一个可供对比的文本表示，以及一个多模态文本解码器，同时完成**图文对比**和**生成 caption**。读代码时问两个问题：哪些向量进入对比损失？哪些 token 进入生成解码器？

```text
图像 → 视觉塔 ──┬→ image_features ─┐
                └→ image_embs      │
文本 → 文本塔 ──┬→ text_features ──┼→ 对比损失
                └→ token_embs ─────┼→ 多模态解码器 → caption 损失
                     image_embs ────┘
```

## 二、只看与 CoCa 直接有关的文件

| 文件 | 作用 |
|---|---|
| [src/open_clip/coca_model.py](../open_clip-CoCa/src/open_clip/coca_model.py) | `CoCa` 类、视觉/文本/解码器三部分、前向和生成 |
| [src/open_clip/loss.py](../open_clip-CoCa/src/open_clip/loss.py) | `ClipLoss`、`CoCaLoss`，把对比与 caption 损失合起来 |
| [src/open_clip/task/coca_task.py](../open_clip-CoCa/src/open_clip/task/coca_task.py) | 训练 batch、预测目标和两种损失输入 |
| [src/open_clip/factory.py](../open_clip-CoCa/src/open_clip/factory.py) | 根据配置名字创建模型、加载权重 |
| [README.md](../open_clip-CoCa/README.md) | CoCa 训练与生成例子 |

`src/open_clip/` 下其他如 SigLIP、CLAP、NaFlex 文件是别的模型，不必为了理解 CoCa 全部读完。

## 三、模型创建：为什么需要三套部件

[CoCa.__init__():130](../open_clip-CoCa/src/open_clip/coca_model.py) 创建 `self.visual`（视觉塔）、`self.text`（文本塔）和 `self.text_decoder`（多模态文本解码器）。`visual` 不只给出整张图的向量，还提供图像 token；`text` 也不只给出整句向量，还给解码器提供 token 表示。原因是：**对比只需整体向量，生成必须逐词访问细节**。

还有一个 `logit_scale`，作用与 CLIP 相同：调节向量相似度进入交叉熵前的尺度。`pad_id/bos_id/eos_id` 是文本序列中的补齐、开始、结束 token ID，关系到目标标签和生成停止条件。

## 四、一次 `forward()` 发生了什么

打开 [CoCa.forward():380](../open_clip-CoCa/src/open_clip/coca_model.py)：

1. 如果输入了图片，`_encode_image()` 生成 `image_latent`（全局）和 `image_embs`（供解码器看的 token）。
2. 如果输入了文字，`_encode_text()` 生成 `text_latent`（全局）和 `token_embs`（供解码器看的 token）。
3. 若只有图片或只有文本，可直接返回单模态表示。正常图文训练两者都有。
4. 字典放入 `image_features`、`text_features`、`logit_scale`，供对比损失使用。
5. `self.text_decoder(image_embs,token_embs)` 生成每个位置对词表的预测分数，供 caption 损失使用。如果传入预处理好的 `labels`，新版实现还能在模型内直接计算 caption CE，减少显存开销。

形状例子：`B=2`、`E=512`、`T=16`、词表有 `V` 个词时，全局向量为 `[2,512]`；文本生成的完整 logits 通常形如 `[2,16,V]`。`V` 可能很大，所以新版代码把部分 loss 计算融合进前向以节省内存。

## 五、`CoCaLoss` 两项为什么都需要

[ClipLoss.forward():118](../open_clip-CoCa/src/open_clip/loss.py) 使用两组全局向量形成图到文、文到图相似度矩阵，鼓励正确配对分数更高。[CoCaLoss.forward():182](../open_clip-CoCa/src/open_clip/loss.py) 在它之上再加 caption 交叉熵，鼓励每个位置正确预测下一个词。两项分别有 `clip_loss_weight`、`caption_loss_weight`，并不是任何时候都必须取 1。

`labels == -100` 的位置不参与词预测损失，通常是 padding。训练时输入的文字与要预测的目标应相差一个位置：例如看到“<BOS> 一只 猫”，目标是“一只 猫 <EOS>”。[CoCaTask._caption_labels():73](../open_clip-CoCa/src/open_clip/task/coca_task.py) 负责构造这类标签。如果输入和目标完全同位，模型可能直接复制当前词，便没有学会生成。

## 六、生成路径怎么读

[generate():72](../open_clip-CoCa/src/open_clip/coca_model.py) 是实际给图片生成文字的入口。流程：编码图片 → 放入起始 token → 解码出下一个词 → 把它接到文本后面 → 重复，直到结束 token 或最大长度。生成不会用正确 caption 做标签。与 `CoCaLoss` 的关系是：训练时学会预测下一个词，推理时反复应用这个能力。

OpenCLIP README 中的 CoCa 例子会创建预训练模型并调用 `model.generate(image)`。这一步可能下载权重；本地目前只有源码，没有模型权重。

## 七、初学者常见误区

- `image_features` 与 `image_embs` 都来自图片，但前者是**一个全局向量**，后者是**多个 token**；用途不同。
- `text_features` 用于对比；生成时不能只拿一个整句向量代替所有 token。
- `CoCaLoss` 是 OpenCLIP 实现中的类名；在论文里关注的是“对比损失＋caption 损失”的思想。
- OpenCLIP 主分支不断扩展，看到不熟悉的额外配置或优化选项先暂时跳过，沿 `CoCa.forward()` 和 `CoCaLoss.forward()` 走通主线。

**本项目完成标准**：能指出四个输出（图像整体、图像 token、文本整体、文本 token）各进入哪一步；能解释为何需要两个损失。
