# 07｜BEiT-3：统一骨干配不同任务头

完整入口：[BEiT-3 论文与代码解释](../UniLM/beit3/解释.md)，包含准确版本 torchscale 源码路径与论文/发布版边界。

> 源码：[UniLM/beit3](../UniLM/beit3)；论文：[BEiT-3.pdf](../papers/BEiT-3.pdf)。这个目录主要展示下游微调、数据和评估。真正的 BEiT-3 主骨干来自外部 `torchscale` 包。

## 一、先理解“统一”是怎么体现的

BEiT-3 把图像、文字以及配对的图文都送到一个 Multiway Transformer 骨干中学习表示。不同任务在同一骨干输出后接不同的任务头：分类要类别分数；检索要图文相似度；VQA 要答案分数；caption 要生成词。对初学者而言，先把**骨干**和**任务头**分开看，比一开始研究所有预训练细节更容易。

```text
图片/文字 → BEiT3 骨干 → 通用特征 → 分类头 / 检索头 / VQA 头 / caption 头
```

## 二、最重要的项目边界

打开 [modeling_utils.py:13](../UniLM/beit3/modeling_utils.py)，能看到它从 `torchscale.model.BEiT3` 导入 `BEiT3`。本地 `modeling_finetune.py` 定义各任务的**包装和输出头**，并没有把骨干所有注意力层直接写在同一文件里。读代码时如果搜不到 backbone 的完整实现，原因在依赖位置，不是自己漏看。

`_get_base_config()` 设置 base 模型的维度、层数、patch 大小；`_get_large_config()` 设置 large 规格。`multiway=True` 告诉底层使用多路模态处理。`BEiT3Wrapper` 创建 `self.beit3=BEiT3(args)`，后续各任务类继承它。

## 三、目录和文件分工

| 文件 | 负责什么 |
|---|---|
| [README.md](../UniLM/beit3/README.md) | 权重、任务与运行说明 |
| [get_started](../UniLM/beit3/get_started) | 分类、检索、VQA、caption、NLVR2 的入门命令 |
| [modeling_utils.py](../UniLM/beit3/modeling_utils.py) | 配置 base/large，包装外部 BEiT3 骨干 |
| [modeling_finetune.py](../UniLM/beit3/modeling_finetune.py) | 五类任务头及其 `forward()` |
| [datasets.py](../UniLM/beit3/datasets.py) | 读取图片、文字、标签、构造索引与 dataloader |
| [engine_for_finetuning.py](../UniLM/beit3/engine_for_finetuning.py) | 通用训练循环及各任务损失/指标处理器 |
| [run_beit3_finetuning.py](../UniLM/beit3/run_beit3_finetuning.py) | 命令行参数和整个微调流程的入口 |

## 四、从一个任务追踪：图文检索

检索问题是“给一张图，找匹配的文字”，反过来也可以给文字找图片。按下面顺序读：

1. [RetrievalDataset.__getitem__():579](../UniLM/beit3/datasets.py) 产生图片、文字 token、padding 信息以及对应的图像 ID。图像 ID 用来知道哪些图文配对正确。
2. [BEiT3ForRetrieval.forward():244](../UniLM/beit3/modeling_finetune.py) 接受图片或文字及 mask，调用 `self.beit3(...)` 得到特征，再经过投影/归一化形成可比较向量。训练时它内部用 `ClipLoss` 算检索损失；`only_infer=True` 时可只取某一侧的特征，方便评估时缓存。
3. [RetrievalHandler.train_batch():113](../UniLM/beit3/engine_for_finetuning.py) 调用模型并把模型给出的损失交给通用训练循环；`eval_batch()`/`after_eval()` 收集特征并计算检索评估结果。
4. [train_one_epoch():454](../UniLM/beit3/engine_for_finetuning.py) 调用处理器、反向传播、更新优化器。这是各任务共用的外层循环。

如果看见 `handler`，可以把它理解成“任务适配器”：同样的训练循环，换一个 handler 就改成另一个任务的输入、损失和评估方式。

## 五、再看视觉推理：为何把 batch 翻倍

[BEiT3ForVisualReasoning.forward():83](../UniLM/beit3/modeling_finetune.py) 用在 NLVR2：输入两张图 `image_a/image_b` 与同一句话。代码把两张图沿 batch 维拼接，文字也复制一份，因此一次让骨干处理 `2B` 个“图＋文”组合。骨干输出包含视觉 CLS 和语言 CLS；先把每个组合的两种 CLS 拼起来，再把属于同一原始样本的 `a`、`b` 两个组合拼起来，形成一个综合向量，最后交给两层 MLP 判断真或假。

形状练习：若每个视觉/语言 CLS 都是 `[B,D]`，每张图与文字先拼成 `[B,2D]`，两张图的结果再拼成 `[B,4D]`，这正好解释 [TwoLayerMLP](../UniLM/beit3/modeling_finetune.py) 的输入为什么设为 `embed_dim * 4`。

## 六、其余任务头如何读

- [BEiT3ForImageClassification:106](../UniLM/beit3/modeling_finetune.py)：只给图片，取全局表示送入分类头。
- [BEiT3ForVisualQuestionAnswering:191](../UniLM/beit3/modeling_finetune.py)：给图片和问题文字，输出候选答案分数。
- [BEiT3ForCaptioning:133](../UniLM/beit3/modeling_finetune.py)：图片加文字前缀，生成 caption 的下一个词；训练时有目标词，推理时逐词生成。

[engine_for_finetuning.py](../UniLM/beit3/engine_for_finetuning.py) 中的 `ImageNetHandler`、`VQAHandler`、`CaptioningHandler` 分别知道各任务该怎么算 loss、该如何报告指标。不要把 `Handler` 误认为新的 Transformer 层；它是训练逻辑。

## 七、初学者最值得记住的事

这个项目最适合学习**大型模型如何拆成共享骨干、任务头、数据集和训练处理器**。若目标是理解 BEiT-3 预训练里的每一层注意力，还需继续阅读 `torchscale` 包中的 `BEiT3` 定义；当前目录单独不足以涵盖它。权重也没有随 Git 源码下载。

**本项目完成标准**：能从 `run_beit3_finetuning.py` 顺着检索任务走到 dataset、model、handler、train loop；能解释为何 `modeling_utils.py` 只见 `BEiT3` 导入而看不到骨干所有层。
