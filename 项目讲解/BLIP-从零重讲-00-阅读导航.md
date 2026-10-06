# BLIP 从零重讲：先看这里

这套讲解按“只有 Python 基础、第一次读深度学习代码”来写。原仓库在 [`BLIP`](../BLIP)，讲解以本机现有代码为准。**先读本页，再按 01→04 顺序读。**每一章把英语缩写、变量名的来源、关键库函数的输入输出、张量形状和代码调用关系写在一起。遇到看不懂的语句，可以对着右侧源码逐项查，不需要先背公式。

| 顺序 | 文件 | 解决的问题 |
| --- | --- | --- |
| 0 | [本页](BLIP-从零重讲-00-阅读导航.md) | 全局地图、符号、读代码方法 |
| 1 | [数据和 ViT](BLIP-从零重讲-01-数据与视觉编码.md) | 图片/文字怎么进来，图片如何变成一串向量 |
| 2 | [MED 和生成](BLIP-从零重讲-02-MED与生成.md) | 文字如何编码、如何看图、如何预测下一个词 |
| 3 | [预训练三目标](BLIP-从零重讲-03-预训练逐步推演.md) | `BLIP_Pretrain.forward()` 每个变量怎么变、三个 loss 如何算 |
| 4 | [下游任务与库函数](BLIP-从零重讲-04-任务与API.md) | 描述生成、检索、VQA 的代码入口及常见调用 |
| 补充 | [已经完成的 `pretrain.py` 逐行代码](BLIP-pretrain-零基础逐行详注.py) | 训练入口每个步骤的中文代码注释 |
| 补充 | [变量与库函数索引](BLIP-pretrain-变量与库函数索引.md) | 训练入口的名字、API、易错点 |
| 源码详注 | [预训练模型教学副本](BLIP-blip_pretrain-零基础逐行详注.py) | 在真实 `blip_pretrain.py` 语句旁补充变量来源、形状和库函数注释 |
| 源码详注 | [ViT 教学副本](BLIP-vit-零基础逐行详注.py) | 图片切块、QKV、多头注意力的就地注释 |
| 源码详注 | [MED 教学副本](BLIP-med-零基础逐行详注.py) | BERT 嵌入、跨注意力、mask、下一词目标的就地注释 |
| 源码详注 | [数据集教学副本](BLIP-pretrain_dataset-零基础逐行详注.py) | 单条图片与文字如何读取和处理 |
| 源码详注 | [通用模型和描述生成教学副本](BLIP-blip-零基础逐行详注.py) | 三种特征模式、caption 训练与生成、权重载入 |
| 源码详注 | [检索教学副本](BLIP-blip_retrieval-零基础逐行详注.py) | 多正例 ID、队列、跨 GPU 难负例 |
| 源码详注 | [VQA 教学副本](BLIP-blip_vqa-零基础逐行详注.py) | 看图问答、候选答案排名 |

## 一句话理解 BLIP

给一张图片和一句话，BLIP 有三种本领：

1. **比较整张图和整句话是否相近**：把它们各压成一个向量，训练图文对齐。源码叫 `loss_ita`；论文常写 ITC（Image-Text Contrastive，图文对比）。
2. **仔细判断这句话是否真的描述了这张图**：让文字每个位置读取图片的局部信息，再分类“匹配/不匹配”。叫 ITM（Image-Text Matching，图文匹配）。
3. **看图续写句子**：给定开头和图，逐词预测后面的词。叫 LM（Language Modeling，语言建模）。

这三件事在 [`models/blip_pretrain.py`](../BLIP/models/blip_pretrain.py) 的同一个 `forward()` 中进行。视觉部分在 [`models/vit.py`](../BLIP/models/vit.py)，文字及跨模态部分在 [`models/med.py`](../BLIP/models/med.py)。

## 先统一记号

| 记号 | 意思 | 例子 |
| --- | --- | --- |
| `B` | Batch size，一批有几对图文 | `B=2` |
| `H,W` | 图片高、宽 | `224,224` |
| `P` | 图像 token 数，含第一个 CLS | `224/16=14`，故 `P=14×14+1=197` |
| `T` | 文字 token 数，含特殊 token 和 PAD | 预训练固定 `T=30` |
| `Dv` | 视觉 token 维度 | ViT Base 是 `768` |
| `Dt` | 文字 token 维度 | BERT Base 是 `768` |
| `E` | 对比学习的共同向量维度 | 默认 `256` |
| `Q` | 历史特征队列容量 | 默认 `57600` |

**张量形状按维度顺序读。**例如 `[B,P,Dv]` 不是“一个大数字”，而是 `B` 张图，每张 `P` 个 token，每个 token `Dv` 个数。`[B,E]` 是每张图只保留一个整体向量。`[:,0,:]` 表示“全部样本、第 0 个 token、该 token 的全部特征”；这里第 0 个就是 CLS。

## 把一次训练串起来

```text
JSON 标注 image 路径 + caption 字符串
    ↓ pretrain_dataset.__getitem__：打开图片、裁剪/增强、清理文字
DataLoader：image [B,3,224,224]，caption 为 B 个字符串
    ↓ BLIP_Pretrain.forward(image, caption, alpha)
ViT(image) → image_embeds [B,197,768]
Tokenizer(caption) → input_ids [B,30]、attention_mask [B,30]
MED 纯文本模式 → text_output [B,30,768]
    ├─ 取两边 CLS → 线性投影/归一化 → ITA 对比损失
    ├─ MED 图文模式：文字看图 → ITM 匹配损失
    └─ MED 解码模式：看图预测下一词 → LM 生成损失
三项 loss 相加 → backward() → optimizer.step()
```

上面 `197` 和 `768` 是 **ViT Base、224 像素、16 像素 patch** 的例子。改图片尺寸或模型大小，数字会变；变量字母表示的含义不变。

## 为什么代码里既有 encoder 又有 decoder

- `visual_encoder`：图像编码器。输入像素，输出图像 token。
- `text_encoder`：文字编码器。`mode='text'` 时只看文字；默认图文模式并传入图像时，还通过交叉注意力看图片。
- `text_decoder`：文字解码器。因果掩码保证当前词只能看此前词；同时能看图片，用于生成。
- 后缀 `_m`：momentum（动量）教师版本，例如 `visual_encoder_m`；通过学生参数的滑动平均更新，不由优化器直接更新。

**同名变量随作用域变化。**`image` 在 Dataset 里是单张图片，进入 `forward()` 后是一批图。`text` 在模型里是 tokenizer 返回的对象，内含 `input_ids` 和 `attention_mask`，不是原来的字符串列表。各章在变量首次出现及隔段再出现处都标明来源。

## 初学者的阅读法

先只看每章的“输入→处理→输出”和形状表。第二遍打开源码，遇到 `x[:,0,:]`、`@`、`.view()` 之类操作就查对应段落。第三遍用两张图、两句文字的纸上例子推一遍 `B=2` 的维度。**不要直接运行官方完整预训练**：默认队列、预训练权重和数据规模都偏大，阅读与小型复现是两个步骤。这里的讲解针对仓库现有代码，未声称已跑通完整预训练。

源码教学副本保留了对应官方文件的可执行语句，新增中文注释；章节正文负责解释跨越多行的原理和变量生命周期。遇到 `med.py` 中与主路径无关的兼容分支，可先跳过。每个教学副本的开头写了它对应的原文件。
