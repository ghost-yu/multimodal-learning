# 第 2 章：MED 如何读文字、看图片、生成句子

对照 [`models/med.py`](../BLIP/models/med.py)。MED 是 **Multimodal mixture of Encoder-Decoder**，可理解为“多模态编码器与解码器混合结构”。它不是另一个外部服务，而是仓库里改造过的 BERT 实现。下文按实际调用链解释，约定 `B=2`、文字长度 `T=30`、文字宽度 `Dt=768`、图像 token 数 `P=197`、视觉宽度 `Dv=768`。

## 1. 先认清 `med.py` 的导入

| 导入 | 作用 | 在本文件的例子 |
| --- | --- | --- |
| `math` | 标准数学函数 | `math.sqrt()` 缩放注意力分数 |
| `os`, `warnings` | 路径与警告 | 兼容/辅助代码；阅读前向主线可略过 |
| `dataclasses.dataclass` | 自动生成数据类方法 | 原仓库沿用的导入，主路径不需要先理解 |
| `typing.Optional, Tuple` | 类型提示，不改变运算 | `Optional` 表示可能为 `None`，`Tuple` 表示元组类型 |
| `torch`, `Tensor`, `nn`, `F` | 张量、类型别名、网络层、函数 | `torch.matmul`、`nn.Embedding`、`F.softmax` |
| `CrossEntropyLoss` | 分类交叉熵类 | 逐 token 比较预测词与真实词 |
| `ACT2FN` | Transformers 激活函数名称到函数的映射 | 根据配置选择 GELU 等 |
| `ModelOutput` 和若干 `...Output` | Transformers 的结构化返回类型 | 允许 `outputs.last_hidden_state`、`outputs.loss` |
| `PreTrainedModel` | Hugging Face 模型基类 | 提供 `from_pretrained`、词表扩展等能力 |
| `apply_chunking_to_forward` | 把长序列分段跑 FFN | 控制峰值内存，结果仍拼回去 |
| `find_pruneable_heads_and_indices`, `prune_linear_layer` | 剪掉指定注意力头 | 主训练路径一般不用 |
| `logging` | 库自带日志接口 | `logging.get_logger(__name__)` 建模块日志器 |
| `BertConfig` | BERT 结构参数容器 | 从 JSON 读取隐藏维、层数、头数 |

`from torch import nn` 在此文件重复导入一次，只会让 `nn` 再次绑定同一模块，不会创建另一套参数。`MaskedLMOutput` 等多个输出类虽导入，BLIP 的预训练主路径主要碰到 `BaseModelOutput...` 与 `CausalLMOutput...`。**导入不等于执行。**

## 2. 先把英文名翻译成可读词

| 英文/变量 | 直译 | 在这里为什么这样叫 |
| --- | --- | --- |
| `BertEmbeddings` | BERT 输入向量层 | 把整数 token ID 查表并加入位置 |
| `hidden_states` | 隐藏状态 | 每层当前的 token 向量；“隐藏”是历史术语，不是不可查看 |
| `attention_mask` | 注意力掩码 | 1 表示有效词，0 表示 PAD；进模型后被扩成可加到分数的形式 |
| `encoder_hidden_states` | 编码器输出状态 | **在 BLIP 里通常就是 ViT 输出的图像 token** |
| `encoder_attention_mask` | 编码器端掩码 | 图像 token 通常都有效，所以全 1 |
| `crossattention` | 交叉注意力 | 文字 Query 读取图像 Key/Value |
| `is_decoder` | 是否作为解码器 | `True` 时建立只能读过去的因果文字 mask |
| `past_key_values` | 历史 Key/Value 缓存 | 逐词生成时避免重算前面词的 K/V |
| `logits` / `prediction_scores` | 未归一化分数 | 每个位置对词表每个词的得分，不是概率 |
| `labels` | 标准答案 | LM 中真实的下一词 ID，`-100` 表示不计分 |
| `mode` | 运行方式 | `'text'` 跳过图文交叉注意力；默认 `'multimodal'` 可以读图 |

**长间隔复现提醒：**本章后面每次写 `encoder_hidden_states`，都指前一章 `ViT(image)` 产生的 `image_embeds [B,P,Dv]`，除非明确指出其他来源；`hidden_states` 则指正在某个 BERT 层里流动的**文字**向量 `[B,T,Dt]`。

## 3. Tokenizer 与 `BertEmbeddings`

模型在 [`init_tokenizer()`](../BLIP/models/blip.py) 中用 `BertTokenizer.from_pretrained('bert-base-uncased')` 载入词表。`BertTokenizer` 是类，`.from_pretrained(...)` 是类方法，可能从本地缓存或网络取得 tokenizer 文件。它把字符串切成 token，并把 token 映射为整数 ID。`add_special_tokens({'bos_token':'[DEC]'})` 增加生成起点；`additional_special_tokens` 增加 `[ENC]` 表示图文联合编码。`tokenizer.enc_token_id` 是额外保存的那个整数 ID，方便代码访问。增加 token 后要调用 `resize_token_embeddings(len(tokenizer))`，不然新增 ID 可能超过旧词向量表大小。

`text = tokenizer(caption,padding='max_length',truncation=True,max_length=30,return_tensors='pt')`：`caption` 是 B 个字符串；`padding='max_length'` 补到 30；`truncation=True` 截断超长句；`return_tensors='pt'` 以 PyTorch 张量返回。结果至少包含 `text.input_ids [B,30]` 和 `text.attention_mask [B,30]`。前者是词 ID，后者是有效位标记。调用 `.to(image.device)` 把张量放到图片所在 CPU/GPU，避免“张量位于不同设备”的错误。

`BertEmbeddings.forward()` 中：

1. `input_shape = input_ids.size()` 取得 `[B,T]`；若没有 ID、给了 `inputs_embeds`，才从向量形状推断。
2. `seq_length = input_shape[1]` 是本批 token 长度 `T`。`position_ids = self.position_ids[:, start:end]` 取位置序号；生成时 `past_key_values_length` 使位置从历史长度继续。
3. `self.word_embeddings(input_ids)` 相当于“按整数查可训练词向量表”，`[B,T]→[B,T,Dt]`。`nn.Embedding(vocab_size,hidden_size)` 是层，不是只做普通 Python 字典查找。
4. `self.position_embeddings(position_ids)` 查同维度位置向量；`embeddings += position_embeddings` 逐元素相加。`+=` 在这里更新局部张量；它不是把序列长度增加。
5. `LayerNorm` 按特征归一化；`Dropout` 在训练时随机置零部分值；最后返回 `[B,T,Dt]`。

## 4. 自注意力和交叉注意力只差 K/V 来源

`BertSelfAttention.__init__(config,is_cross_attention)` 创建三个 `nn.Linear`。`config.hidden_size` 是 `Dt`；`config.num_attention_heads` 是头数 `Hh`；`attention_head_size=Dt/Hh`。`query` 总从文字 `hidden_states` 产生。若 `is_cross_attention=True`，`key`、`value` 层的输入宽度是 `config.encoder_width`，也就是 ViT 的 `Dv`；否则宽度是 `Dt`。

`forward()` 中的 `mixed_query_layer = self.query(hidden_states)`：`mixed` 表示尚未拆成各注意力头的总向量，形状 `[B,T,Dt]`。`is_cross_attention = encoder_hidden_states is not None`：这里再次判断**本次调用是否实际传入图像 token**。交叉分支用 `self.key(encoder_hidden_states)` 和 `self.value(encoder_hidden_states)`；纯文本分支用 `self.key(hidden_states)` 和 `self.value(hidden_states)`。

`transpose_for_scores(x)` 先 `.view(B,长度,Hh,每头维度)`，再 `.permute(0,2,1,3)`，变成 `[B,Hh,长度,每头维度]`。Query 的长度是文字的 `T`；跨模态 Key/Value 的长度是图片的 `P`。

```text
自注意力：Q [B,Hh,T,d] × Kᵀ [B,Hh,d,T] → 分数 [B,Hh,T,T]
交叉注意力：Q [B,Hh,T,d] × Kᵀ [B,Hh,d,P] → 分数 [B,Hh,T,P]
```

`torch.matmul(query_layer,key_layer.transpose(-1,-2))` 正是上面的矩阵乘法。`math.sqrt(self.attention_head_size)` 将分数按每头维度缩放。随后把 mask 加到分数、`softmax(dim=-1)` 得到每个可读位置的权重，再乘 V 加权求和。**跨模态的输出仍有 `T` 个位置**：是每个文字 token 获得图像信息，不是把文字长度变成 `P`。`past_key_value` 缓存的来源是生成时历史文字 K/V；看图像的 K/V 每次可由图像 token 构成。

## 5. `BertLayer` 和 `BertEncoder`

`BertLayer` 是一个大层，顺序是文字自注意力 → 可选图文交叉注意力 → FFN（Feed-Forward Network，前馈网络）。`self.attention` 总会执行。`self.crossattention` 是根据配置**创建出来的层**；只有 `mode=='multimodal'` 才调用。因此“有这层”与“本次使用这层”不是一回事。

`self_attention_outputs = self.attention(hidden_states,...)` 返回元组，`attention_output = self_attention_outputs[0]` 取更新后的文字向量。若 `mode=='multimodal'`，先断言 `encoder_hidden_states` 不为空，然后调用 `self.crossattention(attention_output,...,encoder_hidden_states,encoder_attention_mask)`；此处文字向量当 Query，图像 token 当 K/V。`apply_chunking_to_forward(self.feed_forward_chunk,...)` 调用前馈网络，若配置了分块就按段算。`feed_forward_chunk()` 的 `intermediate` 把特征变宽，`output` 再变回原宽并做残差/归一化。无论哪个模式，输出文字形状仍 `[B,T,Dt]`。

`BertEncoder` 用 `nn.ModuleList([BertLayer(config,i) for i in range(config.num_hidden_layers)])` 建多层。`layer_num` / `i` 是**第几层**；`layer_module = self.layer[i]` 是当前层对象。循环把上一层 `hidden_states` 喂给当前层，随后 `hidden_states = layer_outputs[0]`。**这里是同名变量不断更新，代表“流经第 i 层之后的文字表示”**。`use_cache` 为真时收集每层 K/V，供下一次逐词生成。

## 6. `BertModel.forward()` 的 mask 为什么变成负一万

你在 tokenizer 处见到的 `attention_mask` 是 `[B,T]` 的 0/1。自注意力分数却是 `[B,Hh,T,T]`，所以 `get_extended_attention_mask(...)` 先广播形状，再做 `(1.0-mask)*-10000.0`。有效位 `1→0`，加到原分数相当于不变；无效位 `0→-10000`，softmax 后概率近似 0。**这不是把 PAD 的 token ID 改成 -10000，而是修改“能否被注意”的分数。**

若 `is_decoder=False`，文字可双向看，只需 PAD mask。若 `is_decoder=True`，还创建**因果掩码**：第 0 个词只能看第 0 个；第 1 个词看第 0、1 个；第 2 个看第 0、1、2 个。它与 PAD mask 相乘，所以未来词和 PAD 都不能看。`past_key_values` 存在时，历史缓存区被加到可读位置上。

`BertModel.forward()` 中，`input_ids`、`inputs_embeds`、`encoder_embeds` 是几种输入路径，通常从 `input_ids` 开始。`self.embeddings(...)` 生成文字向量；`self.encoder(...)` 逐层处理。若上游传入 `encoder_hidden_states=image_embeds`，它把图像 mask 通过 `invert_attention_mask` 变成注意力分数可用的形式，并传到各 `BertLayer`。最后 `sequence_output=encoder_outputs[0]` 即 `[B,T,Dt]`。`return_dict=True` 时返回有名字的输出对象，所以外层能写 `text_output.last_hidden_state`。

### 三种调用写法对照

```python
# A：只读文字。mode='text'，每层跳过交叉注意力。
text_output = text_encoder(input_ids, attention_mask=text_mask,
                           return_dict=True, mode='text')

# B：图文融合。传入图像 token，默认 mode='multimodal'。
fused_output = text_encoder(input_ids, attention_mask=text_mask,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True)

# C：看图生成。解码器内部 is_decoder=True，建立因果文字 mask。
decoder_output = text_decoder(input_ids, attention_mask=text_mask,
                              encoder_hidden_states=image_embeds,
                              encoder_attention_mask=image_atts,
                              labels=targets, return_dict=True)
```

A、B 虽用同一个 `text_encoder`，**不是说两次输出相同**。B 中每层多执行了交叉注意力，因此文字表示吸收了图像信息。

## 7. `BertLMHeadModel`：为什么标签要错开一位

`LMHead` 是 Language Modeling Head，即词预测头。`prediction_scores = self.cls(sequence_output)` 把 `[B,T,Dt]` 映射为 `[B,T,V]`，`V` 是词表大小；每个位置对每个可能的下一个词打分。举例输入 `[DEC] 一只 猫 [SEP]`，位置 `[DEC]` 的输出应该预测“一只”，位置“一只”的输出应该预测“猫”。因此源码写：

```python
shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
labels = labels[:, 1:].contiguous()
lm_loss = CrossEntropyLoss(...)(
    shifted_prediction_scores.view(-1, vocab_size), labels.view(-1))
```

`[:, :-1, :]` 去掉最后一个预测位置；`[:,1:]` 去掉第一个目标词，形成“当前预测下一词”的对齐。`.contiguous()` 确保内存排列适合随后 `.view()`；`.view(-1,V)` 把批量和时间合并为若干独立词预测，`-1` 让 PyTorch 推断该维大小。`CrossEntropyLoss` 接收**原始分数**与整数词 ID，无需自己先 softmax。`labels=-100` 的位置自动忽略，例如 PAD。源码还设置 `label_smoothing=0.1`，把“绝对 100% 的正确词”目标稍作平滑。返回对象的 `.loss` 是一个标量。

生成时没有 `labels`，则返回分数并可缓存历史 K/V。`prepare_inputs_for_generation()` 被 `generate()` 每步调用：第一次送整段提示；若已有 `past`，只送最新一个 token，图像 token 从 `model_kwargs` 继续传入。`_reorder_cache` 在 beam search 保留或重排候选句时，同步重排历史缓存，否则文字与缓存会错位。

## 8. 极易混淆的三种 mask/特殊值

| 名字/数值 | 在哪儿 | 作用 |
| --- | --- | --- |
| `attention_mask` 的 `1/0` | tokenizer 输出 `[B,T]` | 1 有效、0 为 PAD |
| 扩展 mask 的 `0/-10000` | 加到注意力分数上 | 0 不影响、-10000 几乎禁止读取 |
| `labels` 的 `-100` | 语言模型交叉熵 | 该位置不算 loss |

这三者含义、使用位置不同。`0` 在 `attention_mask` 中代表不许看，在扩展分数 mask 中代表允许原分数保留；`-100` 是损失函数忽略标记，不是词表里的一个正常 token。

## 9. 阅读边界

`med.py` 是上千行的 BERT 改造文件，包含权重载入、剪注意力头、相对位置编码、输出格式及生成缓存等兼容路径。本章逐步拆解 **BLIP 预训练实际经过的主路径**。如果你跟到 `BertPooler`，本项目常用 `add_pooling_layer=False`，所以主流程读 `last_hidden_state[:,0,:]` 而不是 pooler 输出。`mode='text'`、默认图文模式、解码器模式三条路径先掌握，其他条件分支再查原代码。
