# 04｜BLIP：同一个模型学检索、配对判断和文字生成

完整阅读入口：[BLIP 论文与代码解释](../BLIP/解释.md) · [中文论文](../BLIP/中文论文.pdf) · [详细章节](../BLIP/资料/00-阅读导航.md)。

> 源码：[BLIP](../BLIP)；论文：[BLIP.pdf](../papers/BLIP.pdf)。先读 [ALBEF](03-ALBEF.md)，两份代码有很多相似概念。这里讲的是 **BLIP**，不是 BLIP-2。

**零基础阅读入口**：英文缩写、词组、Python 语法和库函数的逐项解释集中在 [BLIP 术语与库函数详解](04-BLIP-术语与库函数详解.md)。读源码时遇到 `ITA`、`Q/K/V`、`torch.cat` 等，可以直接在那份手册中搜索。

**新版完整精读**：从 [BLIP 从零重讲：阅读导航](BLIP-从零重讲-00-阅读导航.md) 进入，依次读数据与 ViT、MED、预训练三目标、图像描述/检索/VQA，并可打开同目录的源码详注教学副本逐行对照。

针对你粘贴的预训练入口，另有[pretrain.py 零基础逐行详注版](BLIP-pretrain-零基础逐行详注.py)和[变量来历与库函数索引](BLIP-pretrain-变量与库函数索引.md)。详注版会重复提醒相隔较远的变量来源，也标出原稿中容易误读的库函数和未使用的 import。

## 一、项目的核心变化

ALBEF 已经会让图文向量靠近、判断图文配对、根据图片补被遮住的词。BLIP 保留图文对比和匹配，同时加入**从图像生成整句 caption** 的目标。换句话说，它既要会判断“这句话是否描述了图片”，也要能自己说出一句描述。

```text
图片 → ViT → 图像 token ──────────┬→ 对比向量：与文本向量比相似度
                                 ├→ 融合编码器：判断图文是否匹配
                                 └→ 文本解码器：逐词生成图像描述
文字 → tokenizer → 文本 token ───┘
```

## 二、文件分工与阅读顺序

| 文件 | 负责什么 | 推荐先看 |
|---|---|---|
| [pretrain.py](../BLIP/pretrain.py) | 预训练入口 | `train()` 的模型调用和损失合并 |
| [data/pretrain_dataset.py](../BLIP/data/pretrain_dataset.py) | 读图像、caption 与数据标注 | `__getitem__()` |
| [models/blip_pretrain.py](../BLIP/models/blip_pretrain.py) | 三项损失和动量教师 | `forward()` |
| [models/blip.py](../BLIP/models/blip.py) | 通用特征提取、caption 解码与权重加载 | `BLIP_Decoder.generate()` |
| [models/med.py](../BLIP/models/med.py) | 可作文本编码器/解码器的 BERT 变体 | 读完主流程后 |
| [models/blip_retrieval.py](../BLIP/models/blip_retrieval.py) | 图文检索模型 | 选做检索任务时 |
| [models/blip_vqa.py](../BLIP/models/blip_vqa.py) | 视觉问答模型 | 选做 VQA 时 |
| [train_caption.py](../BLIP/train_caption.py) | caption 微调与评估 | 了解生成任务时 |

## 三、预训练的数据怎样进来

[pretrain_dataset.__getitem__()](../BLIP/data/pretrain_dataset.py) 根据标注拿一张图和一条 caption。`pretrain.py` 用 dataloader 合成 batch，传给 `BLIP_Pretrain.forward(image,caption,alpha)`。这里 `caption` 仍是字符串列表，模型内部才 tokenizer，因此不要误以为 `caption` 一开始就已经是 `[B,T]` 整数矩阵。

模型创建见 [BLIP_Pretrain.__init__()](../BLIP/models/blip_pretrain.py)。它准备：

- `visual_encoder`：图片 → patch 特征 `[B,P,Dv]`。
- `text_encoder`：句子 → 文本特征 `[B,T,Dt]`，也能在图像条件下融合。
- `vision_proj/text_proj`：把两边的 CLS 投影到相同 `E` 维，供对比学习。
- `itm_head`：输入融合后特征，输出“匹配/不匹配”两个分数。
- `text_decoder`：看到图像和文字前缀后预测下一个词。
- 带 `_m` 的模型、`image_queue/text_queue`：动量教师与历史特征队列，用法承袭 ALBEF。

## 四、`forward()` 三段拆开看

### 第一段：`loss_ita`

论文称这个目标为 **ITC（Image-Text Contrastive，图文对比）**；仓库使用 `loss_ita` 这个变量名。两者在这里指同一段对比学习代码，详见[缩写词典](04-BLIP-术语与库函数详解.md)。

[BLIP_Pretrain.forward()](../BLIP/models/blip_pretrain.py) 先把图像、文本分别编码成全局向量；用 `F.normalize` 做单位长度化，再计算图到文和文到图的相似度。教师模型提供软目标，队列提供更多候选；两方向交叉熵平均成 `loss_ita`。如果你刚读过 ALBEF，这一段可以对照它的同名变量，重点确认输入是 CLS 全局向量，不是所有 patch 两两相乘。

### 第二段：`loss_itm`

源码 [Image-text Matching 段](../BLIP/models/blip_pretrain.py) 把文本首 token 换成专用于编码的 token，正配对通过 `text_encoder` 访问图像 token。然后根据相似度抽取较难的负图和负文，组成错误配对；`itm_head` 输出真假分数，交叉熵算 `loss_itm`。正例有 `B` 个，通常会构造两类负例，共 `2B` 个，因此标签里常见 `B` 个 1 和 `2B` 个 0。

### 第三段：`loss_lm`

源码 [LM 段](../BLIP/models/blip_pretrain.py) 开始准备 caption 生成训练：

1. `clone()` 复制文本 ID，防止修改上一段仍会用到的输入。
2. 把第一个 token 改为 BOS，意思是“开始说一句话”。
3. `masked_fill(...pad_token_id,-100)` 让 padding 位置不计入损失。
4. `text_decoder` 接收文字 ID 与图像 patch 特征，预测下一词；传入 `labels` 让解码器计算交叉熵。
5. 最后返回 `loss_ita, loss_itm, loss_lm`，训练脚本把它们相加并反向传播。

**生成和补词的区别**：ALBEF 的 MLM 看的是一句话中缺少的词，周围上下文可同时存在；BLIP 的 LM 根据已生成的前文和图像逐步预测后文，因此可在推理时真正生成新句子。

## 五、实际生成 caption 时走哪条路径

[BLIP_Decoder.generate()](../BLIP/models/blip.py) 与预训练 `forward()` 不同：它没有标准答案，必须从 BOS 开始一个词一个词产生输出。`sample=False` 通常走 beam search，`sample=True` 走采样；`max_length` 限制生成长度。`BLIP_Decoder.forward()` 则是有正确 caption 的训练路径，会算 loss。

`blip_decoder(pretrained=...)` 可创建解码模型并加载检查点；`load_checkpoint()` 处理权重来源。第一次读时先看 `generate()` 接收什么、返回什么，不必立刻研究 beam search 的每个内部细节。

## 六、看懂一个下游任务

以图像描述为例：[train_caption.py 的 train()](../BLIP/train_caption.py) 训练时拿图片和 caption，调用模型得到 loss；[evaluate()](../BLIP/train_caption.py) 评估时只给图片，调用 `generate()` 得到新句子。**训练和评估的输入不同**，因为评估时标准 caption 只能用于事后对照，不能喂给生成器当提示而泄露答案。

若改看检索，[blip_retrieval.py](../BLIP/models/blip_retrieval.py) 会先用全局向量粗筛，再用融合匹配细排。若改看 VQA，[blip_vqa.py](../BLIP/models/blip_vqa.py) 则把图片和问题一起编码，对候选答案排名或生成答案。

**本项目完成标准**：能指出三项 loss 使用什么输入；能解释 `text_encoder` 与 `text_decoder` 的不同；知道 `forward()` 训练与 `generate()` 推理的区别。

## 七、第一次读源码：按这条线走

已在原始代码里补充中文行注释。建议先读下面 10 个文件，读完一项再进入下一项；遇到陌生的英文缩写或库函数，先查[术语与库函数手册](04-BLIP-术语与库函数详解.md)。

| 顺序 | 文件 | 先找哪个函数 | 读完应知道什么 |
|---|---|---|---|
| 1 | [configs/pretrain.yaml](../BLIP/configs/pretrain.yaml) | 全部配置项 | 图片大小、batch、队列、学习率从哪里来 |
| 2 | [pretrain.py](../BLIP/pretrain.py) | `main()`、`train()` | 一个 epoch 怎样迭代、求 loss、反向传播 |
| 3 | [data/__init__.py](../BLIP/data/__init__.py) | `create_dataset()`、`create_loader()` | 图片如何增强、样本怎样拼 batch |
| 4 | [data/pretrain_dataset.py](../BLIP/data/pretrain_dataset.py) | `__getitem__()` | JSON 的一条记录如何变成图片张量和字符串 |
| 5 | [models/vit.py](../BLIP/models/vit.py) | `VisionTransformer.forward()` | 图片如何变成 patch token |
| 6 | [models/blip_pretrain.py](../BLIP/models/blip_pretrain.py) | `BLIP_Pretrain.forward()` | ITA、ITM、LM 三个 loss 怎样计算 |
| 7 | [models/med.py](../BLIP/models/med.py) | `BertLayer.forward()` | 文字什么时候自己读文字、什么时候读取图像 |
| 8 | [models/blip.py](../BLIP/models/blip.py) | `BLIP_Decoder.generate()` | 没有标准答案时如何生成图像描述 |
| 9 | [train_caption.py](../BLIP/train_caption.py) | `train()`、`evaluate()` | 预训练后如何微调和评估图像描述 |
| 10 | [utils.py](../BLIP/utils.py) | 三个学习率函数、`MetricLogger` | 学习率与日志统计怎样实现 |

### 从一条 JSON 标注追到三个 loss

假设 JSON 中一项是 `{"image": "cat.jpg", "caption": "a cat on a sofa"}`。`pretrain_dataset.__getitem__()` 读出图片，做随机裁剪、翻转等变换，返回 `[3,224,224]` 张量和一个**字符串**。DataLoader 拼出 `image: [B,3,224,224]` 与 `caption: B 条字符串`。`pretrain.py` 把图片搬到 GPU，再调用 `model(image, caption, alpha)`。

进入 `BLIP_Pretrain.forward()` 后，224×224 图片被 ViT 按 16×16 切成 `14×14=196` 块，加一个 CLS token，得到 `image_embeds: [B,197,768]`。tokenizer 把每条 caption 填充或截断到 30 个 token，得到 `input_ids: [B,30]` 和相同形状的 `attention_mask`。文字经过 BERT 得到 `[B,30,768]`；其中第 0 个 CLS 向量代表整句。图像和文字的 CLS 各经线性层投影成 `[B,256]`，供 ITA 对比学习。

`ITA` 在图像和文字的全局向量之间计算相似度。`image_feat @ text_feat_all` 中的 `@` 是矩阵乘法：`[B,256] @ [256,B+Q] -> [B,B+Q]`。`Q` 是历史特征队列长度，默认 57600。当前批次里第 `i` 张图和第 `i` 句话是真配对。教师模型给出软概率，真配对位置给出硬标签，两者按 `alpha` 混合，产生图找文与文找图两个 loss，平均为 `loss_ita`。

`ITM` 需要更细的图文交互。文字 token 作为 Query，图像的 197 个 token 作为 Key/Value，经过 MED 交叉注意力后才能判断一句话与一张图是否匹配。模型用相似度抽难负例：一批有 `B` 个真配对、`B` 个错图配对、`B` 个错文配对，分类头输出 `[3B,2]`，交叉熵得到 `loss_itm`。

`LM` 用图像 token 作条件，让文字解码器预测下一个词。它把 padding 的目标改为 `-100`，不计入交叉熵。`med.py` 中真正错开一步的位置是：`prediction_scores[:, :-1, :]` 对齐 `labels[:, 1:]`。例如输入前缀“a cat”时，当前位置的输出要预测后面的词。这个损失叫 `loss_lm`。最终 `pretrain.py` 做 `loss = loss_ita + loss_itm + loss_lm`、`loss.backward()`、`optimizer.step()`，参数才改变。

### 读懂三种注意力

ViT 自注意力：图像 patch 之间互相读取，例如猫头 patch 可读取身体 patch。MED 文字自注意力：一句话中的词互相读取。MED 图文交叉注意力：文字提供 Query，图像提供 Key/Value，让词依据图像内容更新。生成 caption 时还会叠加**因果掩码**：预测当前词只能看前面已经生成的词，不能偷看未来答案。

| 符号或代码 | 用白话理解 |
|---|---|
| `B`、`P`、`T`、`D`、`E` | 一批样本数、图像 token 数、文字 token 数、模型隐藏维度、对比学习向量维度 |
| `x[:,0,:]` | 每个样本第 0 个 token 的全部特征，通常是 CLS |
| `dim=0` / `dim=1` | 沿 batch 方向拼 / 沿 token 或候选方向拼；具体还要看当前张量形状 |
| `.detach()` | 得到不参与反向传播的张量视图，常用于教师目标或日志 |
| `with torch.no_grad()` | 此代码块不记录梯度，常用于动量教师和队列操作 |
| `register_buffer()` | 张量随模型保存和搬设备，但不是优化器要训练的参数 |
| `F.normalize(..., dim=-1)` | 把每行向量长度变为 1，使点积可理解成余弦相似度 |
| `.transpose(-2,-1)` / `.t()` | 转置矩阵的两个维度，供 QK 点积或向量匹配 |
| `.contiguous()` / `.view()` | 调整内存布局 / 改变形状，常用于把批量与词位置合并 |

## 八、当前注释范围与运行条件

这次在**预训练入口、数据读取、ViT、MED 关键前向计算、三项目标、caption 生成与微调入口、学习率和日志工具**的原始文件中补了中文行注释。英文缩写、变量名和常见库函数另有[零基础手册](04-BLIP-术语与库函数详解.md)。`med.py` 的权重加载和不常走的兼容分支、检索/VQA 等其他下游任务训练脚本以及第三方增强实现还没有逐行扩写。阅读主流程时可先跳过它们。

仓库的默认 `configs/pretrain.yaml` 中 `train_file` 指向作者机器上的 `/export/...` 路径，本机没有对应数据；模型初始化还可能下载 DeiT/BERT 权重。当前改动只增加解释性注释，已检查 Python 语法，**没有声称完成原版 BLIP 训练**。
