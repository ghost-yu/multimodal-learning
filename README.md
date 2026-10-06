# 多模态经典论文：本地代码与论文索引

## 精读任务与最新入口

[整个任务进度](论文任务进度.md) · [BLIP 完整解释](BLIP/解释.md) · [CoCa 完整解释](open_clip-CoCa/解释.md) · [BEiT-3 详细解释](UniLM/beit3/解释.md)。BLIP、ViLT 中文与双语 PDF 已复制到各项目目录；CoCa 核心路径已扩写注释并通过可执行 AST 核对，极小模拟训练/生成检查成功；其他路径与真实任务实验仍待补齐。

这里有 13 篇论文 PDF，以及 8 个代码仓库。VLMo 和 BEiT-3 共用微软 UniLM 仓库；CoCa 使用 OpenCLIP 的公开实现；PaLI 系列没有在这里对应的原版可运行仓库，放入的是受 PaLI-3 启发的 PaliGemma 参考实现，**不能当成 PaLI 原代码**。

如果是第一次接触多模态代码，**先读 [零基础代码讲解.md](零基础代码讲解.md)**：从 Python/PyTorch 符号和张量形状讲起，对照真实源码逐句解释。之后再读 [代码精读.md](代码精读.md)，把九组论文的入口、模型、损失和数据模块串起来。

现在也有按项目拆开的详细讲解。建议一次只读一篇，打开对应源码跟着变量走：

现有三个可直接运行、带详细中文行注释的教学模型：[CLIP、ViLT 与 ALBEF 小型复现](小型复现/README.md)。CLIP 已用 Flickr8k 小样本完成从零训练，ViLT 与 ALBEF 有合成数据及真实图片训练入口；建议先运行小项目，再看下面的原版仓库讲解。

1. [CLIP：双编码器与相似度矩阵](项目讲解/01-CLIP.md) · [论文与代码解释](Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision/解释.md)
2. [ViLT：图文 token 的早期融合](项目讲解/02-ViLT.md) · [零基础源码精读](项目讲解/ViLT-零基础源码逐行精读.md)
3. [ALBEF：对齐、融合、动量教师](项目讲解/03-ALBEF.md) · [零基础源码精读](项目讲解/ALBEF-零基础源码逐行精读.md)
4. [BLIP：对比、匹配、生成](项目讲解/04-BLIP.md) · [从零重讲与源码详注](项目讲解/BLIP-从零重讲-00-阅读导航.md)
5. [CoCa：对比与 caption 联合训练](项目讲解/05-CoCa.md)
6. [VLMo：共享注意力与模态专家](项目讲解/06-VLMo.md) · [零基础源码精读](项目讲解/VLMO-零基础源码逐行精读.md)
7. [BEiT-3：骨干、任务头和训练处理器](项目讲解/07-BEiT-3.md)
8. [Uni-Perceiver：多任务框架](项目讲解/08-Uni-Perceiver.md)
9. [PaLI：论文与 PaliGemma 代码边界](项目讲解/09-PaLI.md)

## 源码里的中文注释

上述讲解对应的**核心 Python 文件已直接加中文行注释**。打开各篇讲解中链接的 `model.py`、`vilt_module.py`、`model_pretrain.py`、`blip_pretrain.py`、`coca_model.py`、`multiway_transformer.py`、`modeling_finetune.py`、`unified_transformer.py` 或 `paligemma.py`，可在难读的张量操作、分支与损失旁看到解释。训练入口、数据集、CLIP 分词、BLIP 生成/VQA、CoCa 标签移位和部分下游任务代码也已补注释。注释覆盖核心执行路径；辅助脚本和第三方评估工具可依项目讲解中的文件地图继续定位。

| 论文 | 本地 PDF | 本地代码 | 下载时提交 |
|---|---|---|---|
| CLIP | [CLIP.pdf](papers/CLIP.pdf) | [CLIP](CLIP) | `d05afc4` |
| ViLT | [ViLT.pdf](papers/ViLT.pdf) | [ViLT](ViLT) | `762fd39` |
| ALBEF | [ALBEF.pdf](papers/ALBEF.pdf) | [ALBEF](ALBEF) | `b9727e4` |
| VLMo | [VLMo.pdf](papers/VLMo.pdf) | [UniLM/vlmo](UniLM/vlmo) | `31c5b90` |
| BLIP | [BLIP.pdf](papers/BLIP.pdf) | [BLIP](BLIP) | `056a169` |
| CoCa | [CoCa.pdf](papers/CoCa.pdf) | [open_clip-CoCa](open_clip-CoCa) | `2d53460` |
| BEiT-3 | [BEiT-3.pdf](papers/BEiT-3.pdf) | [UniLM/beit3](UniLM/beit3) | `31c5b90` |
| Uni-Perceiver 系列 | [原版](papers/Uni-Perceiver.pdf)、[MoE](papers/Uni-Perceiver-MoE.pdf)、[v2](papers/Uni-Perceiver-v2.pdf) | [Uni-Perceiver](Uni-Perceiver) | `a8c9e37` |
| PaLI 系列 | [PaLI](papers/PaLI.pdf)、[PaLI-X](papers/PaLI-X.pdf)、[PaLI-3](papers/PaLI-3.pdf) | [PaliGemma 替代参考](big_vision-PaliGemma) | `0127fb6` |

## 阅读顺序

1. CLIP：先理解图文编码、归一化、相似度矩阵、双向对比损失。
2. ViLT：理解图像 patch 与文字 token 的早期融合。
3. ALBEF：理解先对齐、再交叉注意力融合，以及动量教师。
4. BLIP：理解对比、匹配、语言生成三目标共用模型。
5. CoCa：在 BLIP 之后比较它的对比加生成方式。
6. VLMo、BEiT-3：看共享注意力与模态专属 FFN。
7. Uni-Perceiver：看统一任务接口；PaLI：看规模化与多语言生成。

## 下载范围与运行说明

- 九组论文对应的原版仓库主要用于读源码；额外在 [小型复现](小型复现/README.md) 中下载了 64 张训练图、16 张验证图与 ViLT 检索权重，用于本机小训练。教学版 ALBEF 与 ViLT 已训练并保存指标。大型模型的全量训练数据并未下载。
- `UniLM` 和 `big_vision-PaliGemma` 使用 Git sparse checkout，只检出了相关目录。`big_vision-PaliGemma` 不是 PaLI 原版。
- 下载成功不等于已跑通原仓库训练。当前已用 `py -3.13` 跑通两个教学模型的真实图片小训练，以及官方 ViLT 权重的检索头微调。多个原仓库依赖特定 PyTorch/CUDA、旧版包或 JAX；若要运行原版 ALBEF 等，需要按各自依赖配置独立环境。
