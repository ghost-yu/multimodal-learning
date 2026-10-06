# ViLT：图片 patch 与文字 token 如何在同一个 Transformer 相遇

源码核对：[vilt_module.py](../ViLT/vilt/modules/vilt_module.py)、[vision_transformer.py](../ViLT/vilt/modules/vision_transformer.py)、[objectives.py](../ViLT/vilt/modules/objectives.py)、[base_dataset.py](../ViLT/vilt/datasets/base_dataset.py)。先读 [概览](02-ViLT.md)。本篇细读数据至预训练目标的主链；其他下游目标请沿 `objectives.py` 函数名继续读。

## 0. 和 CLIP 的结构差在哪里

CLIP 把整图与整句分别编码，最后两个向量相乘。ViLT 在模型较早阶段就把文字 token 和图片 patch 排在一条序列里，令同一层自注意力同时看两种模态。因此输出的 `text_feats` 已经受图片影响，`image_feats` 也已经受文字影响。这里的“融合”是注意力里相互读信息，不是简单地把两个最终向量相加。

下文统一记 `B` 批量、`T` 文字长度、`P` 图片 token 数、`D` 隐藏层宽度、`V` 词表大小。图片通常还有第 0 个 CLS token；精确 `P` 要由实际切块、采样及模型分支决定，不要硬记固定数字。

## 1. 所有核心 `import` 先读懂

| 名字 | 来源及意义 | 在本项目常见用法 |
|---|---|---|
| `torch` | PyTorch 张量库 | `torch.cat` 拼张量；`torch.zeros_like` 做同形状全零张量 |
| `torch.nn as nn` | 可训练神经网络层 | `nn.Embedding(2,D)` 做模态类型查表；`nn.Linear` 做分类 |
| `torch.nn.functional as F` | 函数式操作，不持有参数 | `F.cross_entropy(logits,labels)` 算分类损失；`F.normalize` 做 L2 归一化 |
| `pytorch_lightning as pl` | 训练框架 | `pl.LightningModule` 规定 `training_step`、`configure_optimizers` 等钩子；框架负责循环与设备迁移 |
| `vilt.modules.vision_transformer as vit` | 项目自己的 ViT 实现，`vit` 是局部别名 | `getattr(vit, config['vit'])` 按字符串选构造函数 |
| `BertConfig` | Hugging Face BERT 配置对象 | 用词表数、隐藏宽度、注意力头数等创建文本 embedding 配置 |
| `BertEmbeddings` | Hugging Face 词嵌入层 | 整数 token ID `[B,T]` 转浮点向量 `[B,T,D]` |
| `heads` | 本项目任务头 | `Pooler`、`MLMHead`、`ITMHead` |
| `objectives` | 本项目损失函数集合 | `compute_mlm`、`compute_itm_wpa` 等 |
| `vilt_utils` | 训练配置辅助工具 | `set_task` 选择任务、`set_schedule` 建优化器和学习率计划 |

`objectives.py` 还导入 `os/glob/json`（文件与结果）、`tqdm`（进度条）、`functools`（高阶函数/缓存）、`DistributedSampler` 与项目的 `all_gather`（多卡数据/结果汇总）、`einops.rearrange`（按维度名字重排张量）。这些主要用于完整训练/评估，不是 `infer()` 的必需概念。`import` 只是把名字带进当前文件，不会自动训练模型。

## 2. `__init__()`：模型的零件怎么装

位置：[ViLTransformerSS.__init__](../ViLT/vilt/modules/vilt_module.py)。`config` 是配置字典；写 `config['hidden_size']` 按键取值。`self.hparams.config` 来自 Lightning 的 `save_hyperparameters()`：将构造参数留在模型里，后面 `infer()` 等函数继续读取。名字 `hparams` 是 hyperparameters（超参数）。

1. `BertConfig(...)` 只创建**配置**，不等于下载 BERT 模型。`hidden_size` 是 D；`num_attention_heads` 是多头数；`intermediate_size` 是前馈层宽度。
2. `self.text_embeddings = BertEmbeddings(bert_config)` 创建文字嵌入层。`.apply(objectives.init_weights)` 递归访问子模块并初始化参数；`apply` 不是执行推理。
3. `self.token_type_embeddings = nn.Embedding(2,D)` 建两个可训练向量：索引 0 表示文字，索引 1 表示图像。`token_type` 是“token 属于哪种输入”的类型，不是词性。
4. `getattr(vit, config['vit'])(...)`：`getattr(模块, '函数名')` 从模块中找到同名函数，再用第二个 `(...)` 调它。配置决定 ViT 具体规格。`pretrained=True` 会按实现加载相应的视觉初始化权重；能否成功也取决于权重是否可获取。
5. `self.pooler = heads.Pooler(D)` 把第 0 个融合 token 转成整对图文向量。`mlm_score` 是预测词表的头，`itm_score` 是匹配二分类头，条件 `loss_names[...] > 0` 决定是否创建。
6. 若配置有 `load_path`，`torch.load(path,map_location='cpu')` 读已有 checkpoint，`self.load_state_dict(state_dict,strict=False)` 将匹配的参数装入模型。`strict=False` 容许任务头不同等缺失或多余键；不代表所有结构问题都安全。

## 3. `infer()`：按原代码一段一段读

位置：[ViLTransformerSS.infer](../ViLT/vilt/modules/vilt_module.py)。`infer` 是 inference 的简写，此处既可用于训练也可用于验证；它只抽取特征，损失在外层目标函数算。

### 3.1 从 `batch` 字典找输入

```python
if f'image_{image_token_type_idx - 1}' in batch:
    imgkey = f'image_{image_token_type_idx - 1}'
else:
    imgkey = 'image'
do_mlm = '_mlm' if mask_text else ''
text_ids = batch[f'text_ids{do_mlm}']
text_labels = batch[f'text_labels{do_mlm}']
text_masks = batch['text_masks']
```

`batch` 由 dataloader 和 [`BaseDataset.collate`](../ViLT/vilt/datasets/base_dataset.py) 整理，像普通 Python 字典一样按字符串键查找。`f'...'` 是格式字符串，会把花括号表达式的结果插入字符串。例如 `image_token_type_idx=1` 得到 `image_0`；若字典没有这个键，退回 `image`。它支持 NLVR2 等有多张图的任务。`imgkey` 是 image key 的缩写，存的是**键名字符串**，不是图片。`do_mlm` 用来决定是否取遮住词的版本；`text_ids_mlm` 是模型输入，`text_labels_mlm` 是用于算损失的正确答案。`text_masks` 中真实 token 为 1，padding 为 0。`text_labels` 与 `text_ids` 角色不能混淆。

### 3.2 嵌入图片与文字

`text_embeds = self.text_embeddings(text_ids)`：`embeds` 是 embeddings 的简称，意思是连续向量；输入 `[B,T]` 的整数，输出 `[B,T,D]` 的浮点数。`img = batch[imgkey][0]` 中 `[0]` 取图片列表里的第一个视图；这里的 `img` 是图片张量，不是第 0 张图片。随后 `self.transformer.visual_embed(img,max_image_len=...,mask_it=mask_image)` 返回四项：`image_embeds` 图像 token 向量、`image_masks` 有效 token 标记、`patch_index` 被选择的 patch 原索引、`image_labels` 被遮 patch 任务标签。`mask_it` 不是 ITM 的缩写，意思是“是否对图片做 mask”。

若调用方已给出 `image_embeds/image_masks`，代码不再重复切图，`patch_index,image_labels` 设为 `None`。`None` 是 Python 的“没有值”，不是零张量。此分支可在复用图片特征时节省计算。

### 3.3 加类型、拼序列

`torch.zeros_like(text_masks)` 做 `[B,T]` 全零整数索引，查 `token_type_embeddings` 得 `[B,T,D]` 文字类型向量。`torch.full_like(image_masks,image_token_type_idx)` 做 `[B,P]` 全为图像类型索引。两个向量分别与文字/图片 embedding 相加。类型 embedding 不改变形状，但让模型知道输入来源。

```text
text_embeds  [B,T,D] ──┐
                      ├─ torch.cat(...,dim=1) → co_embeds [B,T+P,D]
image_embeds [B,P,D] ─┘
text_masks   [B,T] ───┐
                      ├─ torch.cat(...,dim=1) → co_masks  [B,T+P]
image_masks  [B,P] ───┘
```

`co` 指共同的 combined/multimodal 序列。`dim=1` 是 token 个数这一维；如写 `dim=-1` 就会错误地拼特征宽度。`co_masks` 必须与 `co_embeds` 顺序相同，否则注意力会把真 token 当作 padding，或反过来。

### 3.4 一层层注意力，再拆开

`x = co_embeds` 使 `x` 初始指向拼好的序列。`for i, blk in enumerate(self.transformer.blocks)`：`enumerate` 给每层编号 `i`，`blk` 是本轮 Transformer block 对象；调用 `blk(x,mask=co_masks)` 让图文 token 在注意力中交互，返回更新后的 `x` 及注意力对象 `_attn`。下划线开头表示后者本函数暂不用。循环后，`x` 仍是 `[B,T+P,D]`，但内容已多次更新，不再等于最初的 `co_embeds`。

`self.transformer.norm(x)` 做末层 LayerNorm。`text_feats=x[:,:text_embeds.shape[1]]` 用原文字长度 `T` 截前半；`image_feats=x[:,text_embeds.shape[1]:]` 取余下图像 token。`cls_feats=self.pooler(x)` 取融合后的第 0 token 再经过池化层，常用于 ITM/VQA。`raw_cls_feats=x[:,0]` 是未过 pooler 的版本，二者不能当成完全相同变量。最后 `ret` 字典把这些特征、原始标签和 mask 统一返回；`ret` 是 return 的简写。

## 4. `forward`、`training_step` 与 `loss`

`forward(batch)` 先查看 `self.current_tasks`，按任务名调用 `objectives.compute_mlm`、`compute_mpp`、`compute_itm_wpa`、`compute_vqa` 等。`ret.update(other_dict)` 将各任务返回的键值放进同一个字典。`training_step()` 由 Lightning 对每个 batch 调用：先 `vilt_utils.set_task(self)` 更新任务，再 `output=self(batch)` 触发 `forward()`，最后将输出中**键名含 `loss`** 的张量相加成 `total_loss` 并返回。`total_loss` 是当前 batch 总损失；Lightning 再负责反向传播与优化器步骤。这个命名规则意味着某个返回键若意外含 `loss`，也可能被纳入总和，读代码时要注意。

## 5. 两个主要目标，连着实际代码理解

### MLM：Masked Language Modeling

`compute_mlm` 先 `pl_module.infer(batch,mask_text=True,mask_image=False)`，得到已经看过图像的 `text_feats [B,T,D]`。`mlm_score(text_feats)` 产生每个位置对 V 个词的分数 `mlm_logits [B,T,V]`。`mlm_labels [B,T]` 是正确词 ID，非预测位置用 `-100`。`F.cross_entropy(mlm_logits.view(-1,V),mlm_labels.view(-1),ignore_index=-100)` 把 B 与 T 展平为 `B*T` 个分类题，跳过 `-100`。`view` 要求内存布局可兼容；若以后自己改代码遇到非连续张量，可检查 `.contiguous()` 或用 `.reshape()`。`phase='train' if pl_module.training else 'val'` 选择日志指标，`getattr` 按动态名字查 `train_mlm_loss` 等对象，不改变损失定义。

### ITM + WPA

`compute_itm_wpa` 中 ITM = Image Text Matching，判断一对图文真/假；WPA = Word Patch Alignment，鼓励真配对的词和 patch 更接近。`itm_labels` 先造一半 1、一半 0，再用 `torch.randperm` 随机打乱；为 0 的样本换入 `false_image_0`，使文字与错误图片配对。`zip(bti,bfi)` 同步遍历真图与假图，`enumerate` 还给出样本索引。

`infer` 输出融合后的 `text_feats/image_feats`。`cost_matrix_cosine` 对每个词与 patch 算 `1 - cosine_similarity`，形状 `[B,T,P]`，越小越相似。对 padding 和特殊 CLS 等位置做 mask，是因为它们不代表真实词/图像区域。`ipot` 近似求最优传输矩阵，告诉模型词与 patch 可以如何分配匹配；它是一个**优化对齐的近似计算**，不能解释成模型自己直接输出了逐词人工标注框。`ot_loss=(dist_pos.sum()-dist_neg.sum())/样本数` 鼓励正例对齐距离小、负例距离大，最后代码乘 `0.1`。同时 `itm_score(cls_feats)` 输出真假两类分数，`F.cross_entropy` 算 ITM 损失。`torch.cuda.amp.autocast(enabled=False)` 暂关自动混合精度，以便相关距离计算使用 float32，减小数值不稳定风险。

## 6. 变量追踪表：隔很远再出现仍可找回来

| 变量 | 何处创建 | 后续用途 |
|---|---|---|
| `config` | 模型构造传入字典 | 被 `save_hyperparameters()` 存到 `self.hparams.config`；设置尺寸和任务 |
| `batch` | dataloader 调用 `collate` 后生成 | `infer` 读图、token ID、mask、标签；目标函数可复制并改其中的图 |
| `text_ids` | `batch['text_ids']` 或 `batch['text_ids_mlm']` | 输入 `BertEmbeddings`；仍是整数 |
| `text_embeds` | `self.text_embeddings(text_ids)` | 加类型后与图片拼成 `co_embeds` |
| `image_embeds` | `visual_embed(img)` 或由调用方传入 | 加类型后进入同一 Transformer |
| `x` | 初始等于 `co_embeds`，每层重赋值 | 最终拆出 `text_feats`、`image_feats` |
| `text_feats` | 最终 `x` 的前 T 个位置 | MLM 分类头和 WPA 距离 |
| `cls_feats` | `self.pooler(x)` | ITM、VQA 等整对图文分类 |

## 7. 初学者可以自检的三个问题

1. 若 `text_embeds=[2,8,768]`、`image_embeds=[2,50,768]`，`co_embeds` 是 `[2,58,768]`，不是 `[2,8,1536]`。
2. 为什么 `text_feats` 不是纯文字特征？因为在拆开前已经与图像 patch 一起经过多层自注意力。
3. `mask_text=True` 是否表示删除文字？不是。它选择按 MLM 规则替换部分 token 的 `text_ids_mlm`，正确词仍保存在标签里。

下一篇可读 [ALBEF 逐行精读](ALBEF-零基础源码逐行精读.md)：它把全局图文对齐与融合任务结合。
