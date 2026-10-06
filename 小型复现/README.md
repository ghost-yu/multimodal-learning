# ViLT 与 ALBEF：零基础小型复现

这里有 ViLT、ALBEF 两个使用 16×16 合成图片的**教学用缩小版**；也有利用真实 Flickr8k 小样本训练的 CLIP 教学版。模型、训练和数据处理都有详细中文行注释。

## 文件地图

| 文件 | 作用 |
|---|---|
| [data.py](data.py) | 生成四种图片、文字 token 和正负样本 |
| [vilt_tiny.py](vilt_tiny.py) | ViLT：图像 patch 与文字 token 拼在一起，送入共享 Transformer |
| [albef_tiny.py](albef_tiny.py) | ALBEF：分别编码图文，先算对比损失，再用交叉注意力算匹配损失 |
| [clip_tiny.py](clip_tiny.py) | CLIP：图文双编码器与 `[B,B]` 双向对比损失 |

按项目阅读：[ViLT 逐步讲解](ViLT-逐步讲解.md)；[ALBEF 逐步讲解](ALBEF-逐步讲解.md)；[CLIP 小型训练逐步讲解](CLIP-逐步讲解.md)。

想找原版仓库及真实图片数据，请看 [真实仓库与数据集](真实仓库与数据集.md)，其中提供 Flickr8k 小样本下载脚本。

现已接入真实图片训练：[Flickr8k 小样本训练完整讲解](真实数据训练.md)，运行入口为 [真实数据小训练.py](真实数据小训练.py)。训练完可运行 [画训练曲线.py](画训练曲线.py)，对照训练集和验证集的变化，直观看到小样本过拟合。

进一步学习：[Flickr30k 检索权重的 ViLT 微调](预训练ViLT说明.md)、[通用预训练 ViLT 对照](通用预训练ViLT说明.md)，以及 [两个从零训练模型的 16 选 1 检索比较](比较小模型检索.py)。

## 运行

在本目录的上一级 `多模态` 文件夹打开 PowerShell：

```powershell
py -3.13 -m pip show torch
py -3.13 小型复现/vilt_tiny.py --steps 120
py -3.13 小型复现/albef_tiny.py --steps 250
```

当前电脑的 `py -3.13` 已安装 PyTorch。其他电脑若未安装，请先根据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择适合自己系统的命令。只用 CPU 也可以运行，训练耗时随电脑而变。`--steps 10` 可先做快速试跑。

每个程序最后会打印四张图与四条文字的匹配结果，以及最终准确率。随机初始化的模型每次结果可能不同；运行步数、随机种子和电脑环境会影响结果。

## 先理解数据

图片的形状是 `[B, 3, 16, 16]`：`B` 是一批图片的张数，`3` 是红绿蓝通道。文字是 `[B, 4]`：四个整数分别代表 `[CLS]`、颜色、形状、`[SEP]`。我们共有四类组合：红方、红圆、蓝方、蓝圆。

训练时，同一类图文是**正例**（标签 1）；把文字错开一位，就得到**负例**（标签 0）。由于图片像素和文字都由程序生成，你可以只关注模型代码，不必先处理真实数据集格式。

## 项目 1：ViLT

按 [vilt_tiny.py](vilt_tiny.py) 顺序读：`PatchEmbed` → `TinyViLT.forward` → `train`。图像被分为 16 个 patch；文字有 4 个 token。再放入 1 个 `[CLS]` 向量，送入 Transformer 的序列长度就是 `1 + 4 + 16 = 21`。`[CLS]` 的最终表示交给分类器，判断图文是否一致。

```text
图片 [B,3,16,16] → 16 个 patch [B,16,D] ┐
文字 [B,4]      → 4 个词向量 [B,4,D]   ├→ 拼接 [B,21,D] → Transformer → 匹配分数 [B,2]
可学习 CLS       → [B,1,D]             ┘
```

这是 **ViLT 的核心“早期融合”路径**。原论文还使用更大的视觉骨干、更多预训练任务与真实数据；本项目只训练图文匹配。

## 项目 2：ALBEF

按 [albef_tiny.py](albef_tiny.py) 顺序读：`encode_image`/`encode_text` → `contrastive_loss` → `matching_logits` → `train`。这里的 `D=32` 是隐藏特征维度。

```text
图片 → 图像编码器 → 图像全局向量 ┐
                                 ├→ 相似度矩阵 [B,B] → ITC 对比损失
文字 → 文字编码器 → 文字全局向量 ┘

图像 patch ＋ 文字 token → 交叉注意力 → ITM 图文匹配损失
```

ALBEF 原文还有动量教师、特征队列、掩码语言建模等。这个版本**仅复现“先对齐，再融合”两条核心计算路径**，不代表论文完整复现。四类小数据上的准确率也不能和论文基准成绩比较。

## 两者区别

ViLT 从一开始就把图文 token 放在同一个 Transformer 里。ALBEF 先用两个编码器产生可比较的图文向量，再让文字通过交叉注意力读取图像 patch。你可以在两份代码里搜索 `torch.cat`、`@`、`MultiheadAttention`，分别找到“拼接”“全局对齐”“细粒度融合”的关键行。

## 阅读顺序与练习

1. 先运行 `--steps 10`，确认入口、输出和标签的含义。
2. 阅读 `data.py`，用笔记下每个张量的形状。
3. 阅读 ViLT 的 `forward`，对着上面的流程图走一遍。
4. 阅读 ALBEF 的两个损失，重点区分 `[B,B]` 相似度矩阵与 `[2B,2]` 匹配分类分数。
5. 自己改 `--steps` 或 `--lr`，观察训练损失与准确率如何变化。

原版源码详解另见 [ViLT](../项目讲解/02-ViLT.md) 与 [ALBEF](../项目讲解/03-ALBEF.md)。
