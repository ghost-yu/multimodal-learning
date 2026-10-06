# 02｜ViLT：图像 patch 和文字 token 一起进 Transformer

建议从实际项目目录的 [ViLT 论文与源码解释](../ViLT/解释.md) 开始，再沿其中链接阅读已添加中文注释的原仓库文件。

> **从零开始读源码**：建议配合 [ViLT 零基础源码精读](ViLT-零基础源码逐行精读.md) 和 [带注释的主模块代码副本](四项目教学源码/ViLT-vilt_module-详注.py)。其中解释了 batch 字典、变量来源、库函数与图文融合的每步形状。

> 源码：[ViLT](../ViLT)；论文：[ViLT.pdf](../papers/ViLT.pdf)。先读完 [CLIP](01-CLIP.md) 再读本项目，比较“后期向量相乘”和“早期 token 交互”。

## 一、它想解决什么

有些早期视觉语言模型先用目标检测器抽取图中物体区域特征，步骤重。ViLT 直接把图像切 patch，和文字 token 一起送入 Transformer。这样模型能在注意力层里直接学习“文字中的 dog 对应哪些图像 patch”。

```text
图片 → patch embedding ┐
                    ├→ 拼接序列 → 同一个 Transformer → 融合特征 → 任务头
文字 → BERT embedding ┘
```

CLIP 的图片和文字各自编码，最后比较两个全局向量；ViLT 在编码阶段就将图文 token 拼接，因此叫**早期融合**。

## 二、项目文件如何分工

| 文件 | 工作 |
|---|---|
| [run.py](../ViLT/run.py) | 根据配置启动 PyTorch Lightning 训练或测试 |
| [vilt/config.py](../ViLT/vilt/config.py) | 模型尺寸、任务开关、批量大小、数据路径 |
| [vilt/datasets/base_dataset.py](../ViLT/vilt/datasets/base_dataset.py) | 从存储数据里取一张图、一段文字及负例 |
| [vilt/datamodules/multitask_datamodule.py](../ViLT/vilt/datamodules/multitask_datamodule.py) | 将一个或多个数据集包装成 dataloader |
| [vilt/modules/vision_transformer.py](../ViLT/vilt/modules/vision_transformer.py) | 图像 patch 的 embedding 与 Transformer block |
| [vilt/modules/vilt_module.py](../ViLT/vilt/modules/vilt_module.py) | ViLT 模型，尤其是 `infer()`、`forward()` |
| [vilt/modules/objectives.py](../ViLT/vilt/modules/objectives.py) | MLM、ITM、VQA、检索等目标的损失 |
| [vilt/modules/heads.py](../ViLT/vilt/modules/heads.py) | 把融合特征变成类别或词表分数 |

先看模型；若不知道某个 batch 字段从哪来，再回头看 dataset。`vilt/utils/write_*.py` 是数据预处理工具，第一遍可以不读。

## 三、数据进入模型之前

[BaseDataset.get_suite():141](../ViLT/vilt/datasets/base_dataset.py) 会组合一份样本里要用的图像、文本以及任务可能需要的假图、假文。[BaseDataset.collate():162](../ViLT/vilt/datasets/base_dataset.py) 把多份样本拼成 batch，并按需要构造 MLM 输入。这里的 `collate` 可以理解成“把单件样本整理成一批能送进模型的张量”。

ViLT 的 `batch` 是一个字典，常见键包括 `image`、`text_ids`、`text_masks`、`text_labels`、`text_ids_mlm`。`text_ids` 是文字 token 的整数 ID；`text_masks` 表示哪些位置是真词；`text_labels` 给损失函数提供目标。**同一个张量不同时承担输入和标准答案两个角色**，读目标代码时要分清。

## 四、`infer()` 是整篇论文的核心

打开 [vilt_module.py:110](../ViLT/vilt/modules/vilt_module.py)：

1. `text_ids` 经过 `BertEmbeddings`，得到文字 token 特征 `[B,T,D]`。
2. 图片传给 [visual_embed():557](../ViLT/vilt/modules/vision_transformer.py)，得到图像 patch 特征 `[B,P,D]`、图像有效位和 patch 信息。源码里还可能给图像序列加特殊 token，准确 `P` 以运行时输出为准。
3. `token_type_embeddings` 给文字和图像加不同类型的向量。它帮助共享 Transformer 区分 token 来自哪种模态。
4. `co_embeds = torch.cat([text_embeds,image_embeds],dim=1)`。如果文字有 32 个 token、图像有 197 个 token，拼完就是 `[B,229,D]`。
5. `co_masks` 也按同样顺序拼接，形状 `[B,229]`。注意力用它忽略 padding。
6. `for blk in self.transformer.blocks` 逐层运行 Transformer。这里所有图文 token 同时在一个 `x` 中，文字 token 可以关注图像 token。
7. 最后把 `x` 在文字长度处切开，得到 `text_feats`、`image_feats`；`pooler(x)` 得到整体融合特征 `cls_feats`。

`infer()` 返回的是一个字典，不是单个 tensor。字典里保存的 `text_feats` 可供 MLM 使用，`cls_feats` 可供 ITM 和 VQA 使用。这样同一次模型结构可以支持多种任务。

## 五、`forward()` 为什么还要调 `objectives.py`

[vilt_module.py:186](../ViLT/vilt/modules/vilt_module.py) 的 `forward()` 查看 `current_tasks`，选择计算哪些目标。如果当前没有训练目标，就只返回 `infer()` 的特征；如果开启 `mlm`，调用 `compute_mlm()`；如果开启 `itm`，调用 `compute_itm_wpa()`；如果开启 `vqa`，调用 `compute_vqa()`。训练步骤在同一文件的 `training_step()` 中汇总各项损失，优化器配置见 `configure_optimizers()`。

三个典型目标：

- **MLM**：[compute_mlm():90](../ViLT/vilt/modules/objectives.py) 用 `infer(mask_text=True)` 的文字特征预测原词。`ignore_index=-100` 让没被遮住的位置不计入损失。
- **ITM + WPA**：[compute_itm_wpa():201](../ViLT/vilt/modules/objectives.py) 构造真图文和假图文，`itm_score` 做配对二分类。同时用 OT/IPOT 比较词与图像 patch，鼓励细粒度对齐。`WPA` 可以理解成“词与 patch 的对齐附加目标”，不要把它误认为单独的图像分类任务。
- **VQA**：[compute_vqa():301](../ViLT/vilt/modules/objectives.py) 用融合后的图文信息预测问题答案。输入应是图片加问题，输出是候选答案上的分数。

## 六、一个张量形状练习

设 `B=2,T=32,P=197,D=768`。在 `infer()` 中写下：

```text
text_embeds  [2, 32, 768]
image_embeds [2,197, 768]
co_embeds    [2,229, 768]
co_masks     [2,229]
text_feats   [2, 32, 768]
image_feats  [2,197, 768]
cls_feats    [2,    768]
```

这里的数字是教学例子，实际 `P` 取决于图像大小、patch 大小和特殊 token 数。重点是**拼接维度是第 1 维，不是特征维**。如果改成 `dim=-1`，特征长度会变为 `2D`，序列仍分不开图文，便不是源码的做法。

## 七、第一次读时常见困惑

- `LightningModule` 是训练框架的封装。`training_step()` 返回损失后，Lightning 负责许多常规训练动作；不要因为没看到裸写的 `optimizer.step()` 就认为没训练。
- `[0]` 出现在 `batch['image'][0]`，是该仓库 batch 结构里的图像视图列表，不是只取 batch 第一张图片。要结合 dataset 的 `collate()` 看。
- `false_image_0` 是构造负配对用的另一张图，不是“模型预测错误的图”。

**本项目完成标准**：你能指着 `infer()` 说清图文在哪里拼接、在哪里交互、在哪里拆开；能指出 MLM 和 ITM 各使用返回字典的哪个字段。
