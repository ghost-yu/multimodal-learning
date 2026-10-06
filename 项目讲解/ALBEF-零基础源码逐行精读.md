# ALBEF：对齐、融合、动量教师与三个损失

对应 [预训练入口](../ALBEF/Pretrain.py)、[核心模型](../ALBEF/models/model_pretrain.py)、[配置](../ALBEF/configs/Pretrain.yaml)。先读 [原有概览](03-ALBEF.md)。下面依本地源码的真实执行顺序解释 `forward()` 的每一段关键语句；复杂的 BERT/ViT 内部实现仍可继续在 `models/xbert.py`、`models/vit.py` 追读。

## 0. 名字与形状先统一

ALBEF = **Align Before Fuse**，先做整图与整句的全局对齐，再进行 token 级图文融合。`B` 是每卡 batch 中配对数；`P` 是图像 token 数；`T` 是文字 token 数；`Dv` 是 ViT 宽度；`Dt` 是 BERT 宽度；`E` 是对比学习共同维度；`Q` 是队列长度。`_m` 后缀表示 momentum teacher（动量教师）版本；没有 `_m` 的是正常由梯度更新的学生网络。`feat` 多指取 CLS、投影、归一化后的**全局向量**；`embeds` 多指每个位置的**向量序列**。两者不能混用。

## 1. 本文件的 `import` 每个做什么

| 导入 | 含义 | 这里的典型调用 |
|---|---|---|
| `from functools import partial` | 预先填入函数的一部分参数 | `partial(nn.LayerNorm,eps=1e-6)` 生成一个以后建 LayerNorm 时默认 eps 的构造器 |
| `VisionTransformer` | 本项目图像编码网络 | `self.visual_encoder(image)` 返回图像 token 序列 |
| `interpolate_pos_embed` | 当图片分辨率与预训练权重不同，插值位置参数 | 调整 DeiT checkpoint 的 `pos_embed` 形状 |
| `BertConfig` | BERT 配置的容器 | `from_json_file(path)` 从 JSON 读结构参数 |
| `BertForMaskedLM` | 可以预测被遮词的 BERT 模型 | `from_pretrained` 装已有文字权重；`mode='text'/'fusion'` 由本仓库修改的 `xbert.py` 实现 |
| `torch` | 张量、无梯度上下文、随机采样、分布式通信 | `torch.cat`、`torch.multinomial`、`torch.no_grad` |
| `torch.nn.functional as F` | 函数式计算 | `F.normalize`、`F.softmax`、`F.log_softmax`、`F.cross_entropy` |
| `from torch import nn` | 可训练层与模型父类 | `nn.Module`、`nn.Linear`、`nn.Parameter` |
| `numpy as np`、`random` | 数值与标准库随机相关 | 当前核心 `forward` 主要使用 `torch` 随机函数；不要以为导入就自动设随机种子 |

`torch.no_grad()` 中的操作不构建反向传播计算图。`@torch.no_grad()` 是函数装饰器，表示整个函数都不记录梯度。`register_buffer(name,tensor)` 把张量纳入模型保存和设备迁移，但不会作为可训练参数；本项目的两个队列和指针就是 buffer。

## 2. `__init__` 中四套学生/教师参数

`visual_encoder` 把图片变成 patch 序列，`text_encoder` 把文字变成 token 序列；`vision_proj`/`text_proj` 各自用 `nn.Linear` 把图/文 CLS 映射到同一 E 维。`itm_head=nn.Linear(Dt,2)` 从融合 CLS 得到“假/真”两类分数。`self.temp=nn.Parameter(torch.ones([])*config['temp'])` 建可学习的标量温度；`torch.ones([])` 产生零维标量张量，不是长度为 1 的向量。

后缀 `_m` 的 `visual_encoder_m`、`text_encoder_m`、`vision_proj_m`、`text_proj_m` 与学生结构相同。`self.model_pairs` 把每一组学生/教师放在同一个二元素列表里；`copy_params()` 用 `zip` 同步遍历两个模型的参数并复制初值，再令教师参数 `requires_grad=False`。它仍会更新，但通过 `_momentum_update()` 的滑动平均更新：

```text
新教师参数 = momentum × 旧教师参数 + (1 - momentum) × 当前学生参数
```

`momentum` 接近 1 时教师变化慢，因而目标分布较稳定。队列 `image_queue/text_queue` 的形状为 `[E,Q]`，每一列是一张图或一句话的历史全局特征；`queue_ptr` 是当前写入起点。代码先随机初始化，再沿 `dim=0` 把每列做 L2 归一化。队列是特征缓存，**不是原图片/原句缓存**，所以后面的细粒度融合不能直接使用它。

## 3. 训练入口：`image,text,alpha` 从哪里来

在 [Pretrain.py 的 `train`](../ALBEF/Pretrain.py)，dataloader 给出图片张量 `image` 和文字字符串列表。`tokenizer(text,padding='longest',truncation=True,max_length=25,return_tensors='pt').to(device)` 生成含 `input_ids`、`attention_mask` 的批量编码对象并搬到设备。`input_ids` 是词整数，`attention_mask` 中 1 表示有效词。`alpha` 来自配置；第一轮用 `min(1,i/len(data_loader))` 从小到大暖起，之后用配置固定值。`loss_mlm,loss_ita,loss_itm=model(image,text_input,alpha=alpha)` 按**返回顺序**解包，最后三项相加反传。

此处 `text` 在 `forward` 里是 tokenizer 的编码对象，不是原始 Python 字符串；`image` 是 `[B,3,H,W]` 张量，不是图片文件路径。

## 4. `forward` 第一段：学生全局图文向量

`self.temp.clamp_(0.001,0.5)` 把温度限制在指定区间。末尾 `_` 是原地修改；放在 `no_grad` 内避免把截断作为计算图里的可微操作。

```python
image_embeds = self.visual_encoder(image)                 # [B,P,Dv]，每个图片位置一个向量
image_atts = torch.ones(image_embeds.size()[:-1],
                        dtype=torch.long).to(image.device)  # [B,P]，全部图像位置有效
image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  # [B,E]
```

`image_embeds.size()[:-1]` 把最后的特征维 `Dv` 去掉，只留下 `[B,P]`；`torch.ones` 因而生成全 1 的图像有效位置 mask。`[:,0,:]` 取每张图的 CLS；`vision_proj` 是视觉投影层；`F.normalize(...,dim=-1)` 按最后一维做 L2 归一化。`image_feat` 单数但包含 B 行，一行一个图的全局特征。

文字先 `self.text_encoder.bert(text.input_ids,attention_mask=...,return_dict=True,mode='text')`。`mode='text'` 只编码文字，不做跨模态交叉注意力。`return_dict=True` 让结果可以用 `.last_hidden_state` 取出全部 token 向量 `[B,T,Dt]`。随后 `text_embeds=text_output.last_hidden_state`，取第 0 个 CLS、经 `text_proj` 和 `F.normalize` 得 `text_feat [B,E]`。这时两个 `feat` 可以点积比较，但两个 `embeds` 的内部维度可以不同。

## 5. 第二段：动量教师、队列、软目标与 ITA

`with torch.no_grad(): self._momentum_update()` 后，教师分别编码当前 B 个图/句，得 `image_feat_m/text_feat_m`。`torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)` 把当前 B 列与历史 Q 列接成 `[E,B+Q]`；`.t()` 把 `[B,E]` 转为 `[E,B]`；`.clone().detach()` 复制并切断旧队列与梯度的关系。`text_feat_all` 同理。变量名中的 `all` 表示当前批加队列的候选集合。

`sim_i2t_m=image_feat_m @ text_feat_all / self.temp` 是 `[B,B+Q]`：第 i 行图，对所有文本候选的相似分数。`i2t` = image to text；`t2i` = text to image。`sim_targets=torch.zeros(...); sim_targets.fill_diagonal_(1)` 在**前 B 列**对角线上标真实配对；因为队列历史特征没有这次配对的对角索引，那里保持 0。`fill_diagonal_` 原地修改。

```python
sim_i2t_targets = alpha * F.softmax(sim_i2t_m,dim=1) + (1-alpha) * sim_targets
```

教师概率提供“也许其他候选有一点相似性”的软信息；一热真配对仍提供明确监督。`alpha=0` 时只信一热标签；`alpha=1` 时只信教师。学生 `sim_i2t=image_feat @ text_feat_all / self.temp` 与 `sim_t2i` 用学生当前特征对教师候选打分。`F.log_softmax(...,dim=1)` 将每行分数转为对数概率；乘软标签、逐行求和取负，再对 B 行 `.mean()` 是软目标交叉熵。两个方向平均得到 `loss_ita`。源码叫 **ITA = image text alignment**；它实现的核心是对比式图文对齐。`_dequeue_and_enqueue(image_feat_m,text_feat_m)` 随后把本批教师特征放入环形队列。

### 队列三行不要跳过

`concat_all_gather` 调用 `torch.distributed.all_gather` 收集所有 GPU 的本批特征，再沿 `dim=0` 拼成总批量。`ptr=int(self.queue_ptr)` 读当前写指针。`self.image_queue[:,ptr:ptr+batch_size]=image_feats.T` 将总批写入列区间；`ptr=(ptr+batch_size)%queue_size` 用取模回到队列开头。源码 `assert queue_size % batch_size == 0` 要求队列长度能被**所有进程汇总后的**批量整除。这里还依赖分布式进程组已经初始化；单独直接调用原模型 `forward` 可能在 `all_gather` 处失败。

## 6. 第三段：融合与 ITM 难负例

正例 `output_pos` 通过 `self.text_encoder.bert(...,encoder_embeds=text_embeds,encoder_hidden_states=image_embeds,mode='fusion')` 获得。`encoder_embeds` 是已算出的文字序列；`encoder_hidden_states` 是图像序列；`mode='fusion'` 让文本在交叉注意力中读取图像。`image_atts` 告诉它图像 token 都有效。`output_pos.last_hidden_state[:,0,:]` 是正配对融合 CLS。

ITM 需要假配对。`weights_i2t=F.softmax(sim_i2t[:,:bs],dim=1)` 只取当前批 B 个候选，因为历史队列只有全局特征，没有完整 patch/token 序列可做融合。`fill_diagonal_(0)` 清掉真配对的权重。对于每一段文字，`torch.multinomial(weights_t2i[b],1).item()` 根据相似度概率抽一张**错误**图片索引；对于每张图则用 `weights_i2t` 抽一段错误文字。概率越高越易抽中，故是“难负例”。`torch.stack` 把 B 个选出的张量重新堆成 batch。

`text_embeds_all=[原文字,错误文字]`，`image_embeds_all=[错误图片,原图片]`，两者沿 `dim=0` 组合形成 2B 个错配：第一组“原文字+错图”，第二组“错文+原图”。随后 `output_neg` 对这 2B 对做融合。再拼 `output_pos` 的 B 个 CLS 与 `output_neg` 的 2B 个 CLS 为 `[3B,Dt]`，经 `itm_head` 得 `[3B,2]` 两类分数。`itm_labels` 前 B 项为 1，后 2B 项为 0；`F.cross_entropy(vl_output,itm_labels)` 算 `loss_itm`。

小心：如果 B=1，清掉对角线后没有任何非配对候选，`torch.multinomial` 将无法抽样。这份原始预训练代码的难负例步骤不适合单样本 batch；小型复现要处理这一边界。

## 7. 第四段：MLM 如何遮词

`input_ids=text.input_ids.clone()` 复制输入以便修改；`labels=input_ids.clone()` 保存正确词。`probability_matrix=torch.full(labels.shape,self.mlm_probability)` 给每个位置相同遮盖概率。`mask()` 里 `torch.bernoulli` 按概率逐位置抽样，再让 PAD 和 CLS 永不遮；`targets[~masked_indices]=-100` 使未遮词不进入损失。

对被选中的位置：先以 0.8 概率换 `[MASK]`；在剩下 0.2 中以 0.5 概率换随机词，所以总概率 0.1；最后的 0.1 保留原词。`&` 是布尔逐元素与，`~` 是取反。`torch.randint(vocab_size,input_ids.shape)` 从词表 ID 范围均匀采随机词。`clone()` 重要：否则原 `text.input_ids` 也可能被原地覆盖。

教师 `text_encoder_m(...)` 在无梯度下输出 `logits_m`，`F.softmax(logits_m,dim=-1)` 转为词表软标签。学生 `text_encoder(...)` 同时接 `labels`、`soft_labels`、`alpha`，其内部在 `xbert.py` 混合真实词与教师分布的 MLM 目标，输出 `mlm_output.loss`。这里 `loss_mlm` 的具体内部细节需要继续追 `BertForMaskedLM.forward`，不能把它误说成仅普通交叉熵。

最后实际返回顺序是 `loss_mlm,loss_ita,loss_itm`，与源码计算顺序 ITA→ITM→MLM 不同。这也是为何训练入口必须按返回顺序解包。

## 8. 变量来源复查

| 变量 | 首次来自哪里 | 后面再见到它时记住什么 |
|---|---|---|
| `image_embeds` | 学生 `visual_encoder(image)` | `[B,P,Dv]`，保留各 patch，供 fusion/MLM |
| `image_feat` | 学生 CLS 经 `vision_proj`、归一化 | `[B,E]`，供 ITA 相似度 |
| `text_embeds` | 学生 BERT `mode='text'` | `[B,T,Dt]`，供 fusion/负例 |
| `text_feat` | 学生 CLS 经 `text_proj`、归一化 | `[B,E]`，供 ITA 相似度 |
| `*_m` | 教师的对应网络 | 不走梯度，跟随学生滑动平均 |
| `*_all` | 当前教师特征与队列拼接 | 用于 ITA；不是完整输入数据 |
| `alpha` | `Pretrain.py` 传入 | 软教师目标与硬标签的混合权重 |
| `bs` | `image.size(0)` | 当前卡的样本数，ITM 在本卡 batch 内抽负例 |

建议边读原文件边暂停在每个段末，自己写下当前张量形状。能解释 `feat` 与 `embeds`、`student` 与 `_m`、以及 2B 个错配的配对顺序，就已经掌握了主干。
