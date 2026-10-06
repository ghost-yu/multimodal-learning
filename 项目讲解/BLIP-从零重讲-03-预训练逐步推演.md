# 第 3 章：`BLIP_Pretrain` 从构造到三个损失

对照 [`models/blip_pretrain.py`](../BLIP/models/blip_pretrain.py)。这是 BLIP 预训练的核心。先看构造器存了什么，再按 `forward(image,caption,alpha)` 的**实际执行顺序**逐段推。形状记号见[导航](BLIP-从零重讲-00-阅读导航.md)。为便于心算，假设 `B=2`、`P=197`、`T=30`、`Dv=Dt=768`、`E=256`、`Q=57600`。

## 1. 导入逐项解释

`from models.med import BertConfig,BertModel,BertLMHeadModel`：第一个是配置对象，第二个是文字/图文编码器，第三个是在 BERT 上加词预测头的生成模型。这里的 `models.med` 是**本仓库文件**，不是 PyPI 上名为 med 的包。

`from transformers import BertTokenizer`：Hugging Face 的文字分词器类；不过本文件调用的是 `models.blip.init_tokenizer()` 包装后的版本。`import transformers` 让后面的 `transformers.logging.set_verbosity_error()` 可用；它调低日志输出级别，不改变训练公式。

`import torch`：张量主库；`from torch import nn`：网络层命名空间，如 `nn.Linear`；`import torch.nn.functional as F`：函数式运算，`F.normalize`、`F.softmax`、`F.log_softmax`、`F.cross_entropy` 都从这里来。`from models.blip import create_vit,init_tokenizer,load_checkpoint` 中前两个在本文件使用，`load_checkpoint` 是保留导入，预训练类本身没有调用它。`from typing import List` 只给末尾函数的参数/变量加类型说明，运行时不是一个实际列表。

## 2. 构造器的变量为什么这样命名

`BLIP_Pretrain(nn.Module)` 的 `Pretrain` 表示先在通用图文数据上训练，之后才能迁移到描述、检索、VQA 等任务。`super().__init__()` 先初始化 PyTorch 父类，后面把层放到 `self.xxx` 才会被正确登记。

| 名字 | 展开/来源 | 类型/形状 | 后续用途 |
| --- | --- | --- | --- |
| `med_config` | MED 的 JSON 配置路径 | 字符串 | `BertConfig.from_json_file()` 读取层数等 |
| `vit` | Vision Transformer 规格 | `'base'` 或 `'large'` | `create_vit()` 选择大小 |
| `vision_width` | 视觉 token 宽度 | 整数 `Dv` | 配置交叉注意力及视觉投影 |
| `text_width` | 文字 token 宽度 | 整数 `Dt` | 配置文字投影与 ITM 分类头 |
| `embed_dim` | 共同 embedding 维度 | 整数 `E` | 把图文 CLS 都映射成 `[B,E]` |
| `queue_size` | 队列容量 | 整数 `Q` | 跨 batch 保存候选图文向量 |
| `momentum` | EMA 保留旧教师的比例 | 默认 `0.995` | `_momentum_update()` |
| `alpha` | 软/硬目标混合比例 | 每次 `forward` 传入 | `alpha×教师软目标+(1-alpha)×硬目标` |
| `temp` | temperature，温度 | 可训练单数 `nn.Parameter` | 相似度除以它再 softmax |

`self.visual_encoder, vision_width = create_vit(...)` 产生视觉模型和输出宽度。Base 分支通过 `torch.hub.load_state_dict_from_url(...)` 下载/缓存 DeiT 视觉权重，`checkpoint['model']` 取其中的参数字典，`load_state_dict(...,strict=False)` 把同名参数装入视觉模型并允许有未匹配层。Large 分支用 timm 的预训练装载函数；首次运行可能需要网络和不少存储空间。

`self.tokenizer=init_tokenizer()` 新增 `[DEC]`、`[ENC]`；`encoder_config=BertConfig.from_json_file(med_config)` 读 JSON，紧接着 `encoder_config.encoder_width=vision_width` 告诉文字交叉注意力图像向量多宽。`BertModel.from_pretrained('bert-base-uncased',...)` 载入 BERT 初始参数。`resize_token_embeddings(len(self.tokenizer))` 让词向量表覆盖新增 token。`text_width=self.text_encoder.config.hidden_size` 从真实配置取文字宽度，避免写死 768。

`vision_proj=nn.Linear(Dv,E)` 与 `text_proj=nn.Linear(Dt,E)` 是两侧投影层；`itm_head=nn.Linear(Dt,2)` 输出匹配/不匹配两个原始分数。后缀 `_m` 的 `visual_encoder_m`、`vision_proj_m`、`text_encoder_m`、`text_proj_m` 是 momentum 教师版。`model_pairs` 把四组 `[学生模块,教师模块]` 对齐；`copy_params()` 先把学生参数复制给教师，再冻结教师的梯度。`_m` 不是“多模态”的缩写，在这些变量里是**动量**。

`register_buffer('image_queue',torch.randn(E,Q))` 和 `text_queue` 建特征队列。buffer 会跟着 `model.to(device)` 搬设备、保存到 `state_dict`，但不是 optimizer 要更新的参数。`queue_ptr` 是一个整数张量，记录下一批写到哪里。队列按 `dim=0` 归一化，因为**每列**是一条 `E` 维样本。`self.temp=nn.Parameter(0.07*torch.ones([]))` 中 `torch.ones([])` 是 0 维标量张量；乘 `0.07` 初始化温度，再包成可学习参数。它在前向开始会被限制在 `[0.001,0.5]`。

`text_decoder=BertLMHeadModel.from_pretrained(...)` 创建生成器；最后 `tie_encoder_decoder_weights(...)` 尝试让对应的编码器和解码器层共享同一个参数对象。末尾递归函数在类类型不相等的分支引用未定义的 `logger`；这是仓库源码的潜在问题，阅读时不要把它误认为完整可用的日志逻辑。通常是否触发要看传入的两个具体对象类型。

## 3. `forward` 的输入来源与第一批变量

`image` 来自 DataLoader 的图片批量 `[B,3,H,W]`；`caption` 是 **B 个 Python 字符串**；`alpha` 是训练入口按轮次传来的混合系数。温度限制写成：

```python
with torch.no_grad():
    self.temp.clamp_(0.001,0.5)
```

`torch.no_grad()` 暂停梯度记录；`clamp_` 把标量裁到上下界，末尾 `_` 表示原地修改。它不是把 `temp` 固定为不可训练；其他使用 `self.temp` 的计算仍可对它产生梯度。

接着：

| 语句与变量 | 来源/命名 | 形状和意义 |
| --- | --- | --- |
| `image_embeds = visual_encoder(image)` | `embeds` = embeddings，ViT 的图像 token | `[B,P,Dv]`，此处 `[2,197,768]` |
| `image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)` | `atts` = attention mask；图像 token 全有效 | `size()[:-1]` 去掉特征维，得 `[B,P]`；`ones` 全为 1 |
| `image_feat = F.normalize(vision_proj(image_embeds[:,0,:]),dim=-1)` | `feat` = 整图 feature；取第 0 个 CLS | `[B,E]`；`dim=-1` 按每个样本的 256 个数归一化 |
| `text = tokenizer(caption,...).to(image.device)` | `text` **现已是 tokenizer 输出对象** | `.input_ids` / `.attention_mask` 各 `[B,T]` |
| `text_output = text_encoder(...,mode='text')` | 纯文本 BERT 输出对象 | `.last_hidden_state [B,T,Dt]` |
| `text_feat = F.normalize(text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)` | 整句 CLS 特征 | `[B,E]`，尚未与图片交互 |

`F.normalize(x,dim=-1)` 默认做 L2 归一化，即把最后一维向量长度变为约 1。之后点积近似余弦相似度。`return_dict=True` 让文字模型返回可按 `.last_hidden_state` 访问的结构。`padding='max_length'` 和 `max_length=30` 使本路径文字长度固定为 30。

## 4. 动量教师与历史队列：准备“比较对象”

`with torch.no_grad()` 下面都不用梯度反向传播。`_momentum_update()` 按每个参数做：

```text
教师新参数 = momentum × 教师旧参数 + (1-momentum) × 学生当前参数
```

`momentum=0.995` 意味着教师这一步只吸收 0.5% 学生参数，变化较平滑。之后教师 ViT/文字编码器得到 `image_embeds_m [B,P,Dv]`、`image_feat_m [B,E]`、`text_output_m`、`text_feat_m [B,E]`。此处 `_m` 仍表示“动量教师”。

`image_feat_m.t()` 把 `[B,E]` 转置成 `[E,B]`；`self.image_queue.clone().detach()` 是历史图像候选 `[E,Q]` 的副本并断开梯度；`torch.cat([...],dim=1)` 沿候选维拼成 `image_feat_all [E,B+Q]`。文字同理得到 `text_feat_all [E,B+Q]`。后缀 `_all` 表示“当前批的教师特征 + 过去批次的队列特征”，**不是所有训练数据**。

`@` 是矩阵乘法。`sim_i2t_m = image_feat_m @ text_feat_all / self.temp` 的尺寸是 `[B,E]@[E,B+Q]→[B,B+Q]`。`i2t` = image to text，**每行一张图，每列一个文字候选**；`t2i` 则每行一句话，每列一个图片候选。`sim` = similarity；尾部 `_m` 表示相似度由教师查询算出，用来构造目标。除温度后再 softmax，温度越小分布通常越尖锐。

## 5. ITA：软目标、硬目标、学生损失

`sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)` 新建 `[B,B+Q]` 的全零矩阵。`fill_diagonal_(1)` 把 `(0,0),(1,1)...` 设 1：本 batch 第 `i` 张图对应第 `i` 段文字。**队列列在当前硬标签里全是 0。**

`sim_i2t_targets = alpha*F.softmax(sim_i2t_m,dim=1)+(1-alpha)*sim_targets`：教师相似度的 `softmax` 得到软概率，和真实配对的一热硬标签混合。`dim=1` 是每行跨候选归一化。`sim_t2i_targets` 同理。`alpha=0` 时只用硬标签；`alpha=1` 时只用教师软分布。这里的目标在 `no_grad` 中构造，不对教师反传。

学生用自己的 `image_feat`/`text_feat` 对**同一批候选列**打分：

```python
sim_i2t = image_feat @ text_feat_all / self.temp
sim_t2i = text_feat @ image_feat_all / self.temp
loss_i2t = -torch.sum(F.log_softmax(sim_i2t,dim=1)*sim_i2t_targets,dim=1).mean()
loss_t2i = -torch.sum(F.log_softmax(sim_t2i,dim=1)*sim_t2i_targets,dim=1).mean()
loss_ita = (loss_i2t+loss_t2i)/2
```

`F.log_softmax` 把每行分数变为对数概率；与目标分布逐项相乘，对候选求和、取负，得到每行的软标签交叉熵；`.mean()` 再对 B 张图/句取平均。双方向平均就是 `loss_ita`。**论文常把这一目标叫 ITC（contrastive，对比），这份代码名 `ITA`（alignment，对齐），指同一段图文对齐训练。**

`_dequeue_and_enqueue(image_feat_m,text_feat_m)` 然后把当前**教师**特征写进历史队列。其内部 `concat_all_gather` 先收集所有 GPU 的批特征，所以要求分布式环境已初始化。`ptr=int(self.queue_ptr)` 是下一段写入位置。代码 `assert self.queue_size % batch_size == 0` 强制队列容量能被**全局批量**整除；不满足会断言失败。`image_queue[:,ptr:ptr+batch_size]=image_feats.T` 沿列覆盖；`ptr=(ptr+batch_size)%queue_size` 到末尾后回到开头。`Q` 初始为随机归一化特征，训练一段后逐步被真实教师特征覆盖。

## 6. ITM：为什么要再跑一次文字编码器

ITA 只比较整图/整句 CLS。为了判断细节是否匹配，ITM 让文字逐 token 读取图片 token。`encoder_input_ids=text.input_ids.clone()` 复制整数 ID，随后 `encoder_input_ids[:,0]=tokenizer.enc_token_id` 把每句第 0 个 token 改成 `[ENC]`。`clone()` 保留原始 `text.input_ids` 给后面的 LM 用。`bs=image.size(0)` 就是本批大小 B；源码用小写 `bs`，英文 `batch size`。

正例：`output_pos=text_encoder(encoder_input_ids, attention_mask=..., encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,return_dict=True)`。`encoder_hidden_states` 在这一行明确来自**本次学生 ViT 的 `image_embeds`**，不是教师队列，也不是文字编码器输出。默认 `mode='multimodal'` 使 MED 的交叉注意力生效。结果 `.last_hidden_state [B,T,Dt]`，其中第 0 位汇总了图文信息。

困难负例：`sim_t2i[:,:bs]` 只取本 batch 的图片候选列，因为队列里只有图像**全局向量**，没有完整的 `image_embeds [P,Dv]` 可供交叉注意力。`F.softmax(...,dim=1)+1e-4` 形成非负抽样权重；`fill_diagonal_(0)` 禁止抽到真配对。`torch.multinomial(weights_t2i[b],1).item()` 按权重随机抽一个错误图片下标；`b` 是当前**第 b 条文字**，`neg_idx` 是被抽中的**另一张图片在 batch 里的序号**。用 `image_embeds[neg_idx]` 得到错图完整 token。反方向 `weights_i2t` 为每张真图抽错文字。

`image_embeds_neg` 是 B 张负图 token，`text_ids_neg`、`text_atts_neg` 是 B 段负文的 ID 与 mask。`torch.stack(list,dim=0)` 在最前面新增 batch 维。拼接后的 `text_ids_all=[真文字,错文字]`、`image_embeds_all=[错图片,真图片]`，各 2B 条；它们**按位置配对**，所以前 B 条是“真文字+错图片”，后 B 条是“错文字+真图片”。`output_neg=text_encoder(...)` 一次运行 2B 个错误配对。

`vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:],output_neg.last_hidden_state[:,0,:]],dim=0)` 把正例和负例的融合 CLS 拼成 `[3B,Dt]`。`vl` = vision-language，图文。`vl_output=itm_head(vl_embeddings)` 得 `[3B,2]` 两类原始分数。`itm_labels` 前 B 个 1（匹配），后 2B 个 0（不匹配）。`F.cross_entropy(vl_output,itm_labels)` 内部计算分类交叉熵，得到标量 `loss_itm`。这里 B 必须能抽到非自身负例；若 `B=1`，对角线置零后无可抽候选，原实现会失败。

## 7. LM：图像条件下预测下一词

`decoder_input_ids=text.input_ids.clone()` 再次从 tokenizer 的**原始文字 ID** 拷贝，不使用刚才改成 `[ENC]` 的副本；`[:,0]=tokenizer.bos_token_id` 改成 `[DEC]` 起点。`decoder_targets=decoder_input_ids.masked_fill(decoder_input_ids==pad_id,-100)`：布尔比较找到 PAD，`.masked_fill(...,-100)` 生成把 PAD 位置改为忽略标签的张量。`decoder_input_ids` 仍保存原 token ID，只有 `decoder_targets` 的 PAD 变成 -100。

`text_decoder(...,encoder_hidden_states=image_embeds,encoder_attention_mask=image_atts,labels=decoder_targets)` 中的图像仍是本次学生 ViT 的完整 token。MED 解码器对文字使用因果 mask，又用交叉注意力看图。`BertLMHeadModel` 内部把位置 `0..T-2` 的预测对齐位置 `1..T-1` 的标签，并计算交叉熵。外层 `loss_lm=decoder_output.loss` 是标量。最后 `return loss_ita,loss_itm,loss_lm` 返回三元素元组；训练入口再把它们相加后反向传播，见[训练脚本逐行版](BLIP-pretrain-零基础逐行详注.py)。

## 8. 一张变量“身份证”：隔很久再出现时去哪里找

| 名字 | 最初创建处 | 它到底是什么 | 何处复用 |
| --- | --- | --- | --- |
| `image` | `forward` 参数 | 本批原始图片张量 `[B,3,H,W]` | `.device`、`.size(0)`；本身不进入 ITM 抽样 |
| `caption` | `forward` 参数 | B 个原始文字字符串 | 仅 tokenizer 输入 |
| `text` | `tokenizer(caption,...)` | 含 `input_ids` / `attention_mask` 的输出对象 | ITA、ITM、LM 都从它读文字 |
| `image_embeds` | 学生 `visual_encoder(image)` | 完整图像 token `[B,P,Dv]` | ITM 正负例、LM 条件 |
| `image_feat` | 学生图像 CLS 投影 | `[B,E]`，归一化整图向量 | 学生 ITA 相似度 |
| `image_embeds_m` | 教师 `visual_encoder_m(image)` | 教师完整图像 token | 只用于取教师 CLS |
| `image_feat_m` | 教师图像 CLS 投影 | `[B,E]` | 教师目标、队列入队 |
| `image_feat_all` | 教师当前特征与旧队列拼接 | `[E,B+Q]` 候选矩阵 | 教师和学生 t2i 相似度 |
| `text_feat_all` | 教师当前文字与旧队列拼接 | `[E,B+Q]` 候选矩阵 | 教师和学生 i2t 相似度 |
| `encoder_input_ids` | `text.input_ids.clone()` 后改 `[ENC]` | 图文融合用文字 ID | ITM 正例与负例文本 |
| `decoder_input_ids` | 再次从 `text.input_ids.clone()` 后改 `[DEC]` | 自回归生成用文字 ID | LM 解码器 |

## 9. `copy_params()`、共享权重与实际训练边界

`zip(model_pair[0].parameters(),model_pair[1].parameters())` 把学生/教师参数按迭代顺序配对；`param_m.data.copy_(param.data)` 复制初值；`param_m.requires_grad=False` 不让教师参数走优化器梯度。`_momentum_update` 的 `.data=...` 用 EMA 原地更新教师数值。`@torch.no_grad()` 是装饰器，表示整个函数不记录梯度；其作用与函数体内 `with torch.no_grad()` 相似，但范围是整个函数。

`tie_encoder_decoder_weights` 递归遍历 `nn.Module._modules` 子层，把兼容层的 `encoder_pointer.weight` 直接指向 `decoder_pointer.weight`，这是**共享同一个 Parameter 对象**，不同于 `copy_()` 只复制一次初值。`skip_key='/attention'` 让指定路径不共享；这一函数较复杂，初读不必把递归每个分支背下来，但要能区分“教师复制/EMA”和“编码器解码器共享”。

**运行限制要看清：**官方这段预训练 forward 的队列调用 `torch.distributed.get_world_size()` / `all_gather`，默认需要已初始化进程组；队列大小要整除全局 batch；ITM 难负例要求本地 batch 至少 2。仅复制模型代码并在普通单进程脚本里调用 `forward`，不能保证直接跑通。小型复现可以先用[既有训练入口注释](BLIP-pretrain-零基础逐行详注.py)理解，再单独调整这些约束。
