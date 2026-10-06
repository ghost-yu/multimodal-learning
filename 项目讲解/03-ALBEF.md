# 03｜ALBEF：先让图文靠近，再做细粒度融合

建议从原项目目录的 [ALBEF 论文与源码解释](../ALBEF/解释.md) 开始，并沿其中链接阅读已加中文注释的实际源码。

> **从零开始读源码**：建议配合 [ALBEF 零基础源码精读](ALBEF-零基础源码逐行精读.md) 和 [带注释的预训练模型副本](四项目教学源码/ALBEF-model_pretrain-详注.py)。其中详细追踪 `feat`/`embeds`、教师 `_m`、队列、难负例与三个损失。

> 源码：[ALBEF](../ALBEF)；论文：[ALBEF.pdf](../papers/ALBEF.pdf)。建议读过 [CLIP](01-CLIP.md) 和 [ViLT](02-ViLT.md) 后再看。仓库已归档，代码依然可读。

## 一、为什么会有“Align before Fuse”

假设图像和文字一开始来自两套不同的网络，特征的含义还对不上。ALBEF 先用 CLIP 风格的对比学习，让配对的整张图和整句话靠近；然后再让文字 token 与图像 patch 做交叉注意力，判断细节是否匹配。这样模型同时有**全局对齐**和**局部融合**。

```text
图片 → ViT → 图像序列 → 图像全局向量 ┐
                                     ├→ 图文对比损失 ITA
文字 → BERT 文本模式 → 文本序列 → 文本全局向量 ┘

图像序列 + 文本序列 → BERT 融合模式 → 图文匹配 ITM / 掩码词 MLM
```

ALBEF 还维护一套更新较慢的**动量教师模型**，帮助处理图文数据中可能出现的不准确配对。

## 二、文件分工

| 文件 | 工作 |
|---|---|
| [Pretrain.py](../ALBEF/Pretrain.py) | 预训练入口，建数据、模型、优化器，调用训练循环 |
| [configs/Pretrain.yaml](../ALBEF/configs/Pretrain.yaml) | 图像大小、队列大小、温度、学习率等配置 |
| [dataset/caption_dataset.py](../ALBEF/dataset/caption_dataset.py) | 读图片路径和 caption，产生图文样本 |
| [dataset/__init__.py](../ALBEF/dataset/__init__.py) | 根据任务创建数据集、采样器、dataloader |
| [models/model_pretrain.py](../ALBEF/models/model_pretrain.py) | 三项目标、教师、队列的核心实现 |
| [models/xbert.py](../ALBEF/models/xbert.py) | BERT 文本模式与图文融合模式 |
| [models/model_retrieval.py](../ALBEF/models/model_retrieval.py) | 下游图文检索与重排 |
| [Retrieval.py](../ALBEF/Retrieval.py) | 检索微调与评估入口 |

## 三、从一批数据到 `forward()`

[pretrain_dataset.__getitem__():97](../ALBEF/dataset/caption_dataset.py) 按索引读取一张图及其文字描述。dataloader 将多个样本合成一个 batch。[Pretrain.py 的 `train():35`](../ALBEF/Pretrain.py) 对文字调用 tokenizer，将句子变为 `input_ids` 与 `attention_mask`，随后调用 `model(image, text, alpha)`。这里的 `alpha` 控制教师软目标与真实配对目标的混合权重。

先只看 [ALBEF.forward():88](../ALBEF/models/model_pretrain.py) 的前四个输出：

| 变量 | 含义 | 典型形状 |
|---|---|---|
| `image_embeds` | ViT 每个图像 token 的表示 | `[B,P,Dv]` |
| `image_feat` | 取 CLS、线性投影并归一化后的整图向量 | `[B,E]` |
| `text_embeds` | BERT 文本模式的各词表示 | `[B,T,Dt]` |
| `text_feat` | 取 CLS、线性投影并归一化后的整句向量 | `[B,E]` |

图像向量和文字向量都必须是 `E` 维，后续才能做 `image_feat @ text_feat.T`。`image_embeds` 与 `text_embeds` 仍保留每个 token 的信息，稍后用于融合。

## 四、第一个损失 `loss_ita`：全局对齐

`sim_i2t` 表示每张图片对候选句子的相似度，`sim_t2i` 是反方向。代码不仅比较当前 batch，还把过去 batch 的教师特征放在 `image_queue/text_queue` 里，形成更多候选。若当前 batch 有 `B` 个样本、队列有 `Q` 个向量，相似度大致是 `[B,B+Q]`。

教师模型的变量名带 `_m`，如 `visual_encoder_m`、`text_encoder_m`。`_momentum_update()` 不是梯度下降，而是慢慢把学生参数复制过去，可理解成“过去许多训练步骤的平滑版本”。教师计算 `sim_i2t_m`、`sim_t2i_m`，再经 softmax 给出软分布；代码用 `alpha` 混合软分布和真实配对的一热标签。学生的 `sim_i2t` 和 `sim_t2i` 对这些目标分别算交叉熵，平均后得到 `loss_ita`。

**队列不是图片文件仓库**：里面存的是编码后的向量。`_dequeue_and_enqueue()` 将新向量放进去、把旧向量挤出，节约重新编码历史图片的成本。

## 五、第二个损失 `loss_itm`：融合后判断真假

[model_pretrain.py:133](../ALBEF/models/model_pretrain.py) 开始，真实配对的 `text_embeds` 被送到 BERT 融合模式，`encoder_hidden_states=image_embeds` 使文字能通过交叉注意力读取图像 token。模型也按相似度抽取**难负例**：比如一张狗图配上“草地上的狼”，比配上“一架飞机”更容易迷惑模型。`itm_head` 最终输出真假两类分数，`cross_entropy` 得到 `loss_itm`。

之所以既要 `ITA` 又要 `ITM`：ITA 高效地比较全局向量，适合粗筛；ITM 让 token 深度交互，适合判断细节。检索代码 [model_retrieval.py](../ALBEF/models/model_retrieval.py) 正是利用这两种能力做候选筛选与重排。

## 六、第三个损失 `loss_mlm`：在图像帮助下补词

源码 [model_pretrain.py:187](../ALBEF/models/model_pretrain.py) 复制文字 ID，随机把一部分 token 改成 mask，再让模型在图像条件下预测原词。`mask()` 在同一文件的 [第 253 行](../ALBEF/models/model_pretrain.py)，负责选位置和构造目标。教师也对遮词输入给出词表概率，学生同时参考真实词与教师软分布，最终得到 `loss_mlm`。

`forward()` 返回的顺序是 **`loss_mlm, loss_ita, loss_itm`**。不要因为前面先算 ITA，就以为返回顺序也是 ITA 第一。读训练脚本时要根据解包变量确认。

## 七、如何读训练循环

[Pretrain.py:35](../ALBEF/Pretrain.py) 的每次迭代大致是：取图文 batch → 移到 GPU → tokenizer 处理文字 → `model(...)` 得到三个损失 → 相加 → `backward()` → `optimizer.step()`。`main()` 创建数据集、模型、优化器和学习率调度器，并按 epoch 调用 `train()`。论文规模的预训练需要大量数据和显卡；读懂 `forward()` 不要求先把整个训练跑完。

## 八、第一次读时的三个易错点

1. `mode='text'` 是只编码文字；`mode='fusion'` 才发生图文交叉注意力。它们都调用文本编码器相关模块，但作用不同。
2. `_m` 教师参数由动量更新，通常不参与梯度反传；不要把教师误认为另一个独立训练目标。
3. ITA 的负例来自相似度矩阵；ITM 的负例是具体的错误图文配对，后者还需要再次融合计算。

**本项目完成标准**：你能在 `forward()` 里分别圈出 ITA、ITM、MLM 的代码块，并说清教师模型、队列、`alpha` 各自干什么。
