# 第 4 章：图像描述、检索、VQA 与常用库函数

本章把预训练组件接到实际任务。对应源码：[通用模型/图像描述](../BLIP/models/blip.py)、[图文检索](../BLIP/models/blip_retrieval.py)、[视觉问答](../BLIP/models/blip_vqa.py)、[描述训练入口](../BLIP/train_caption.py)、[检索训练入口](../BLIP/train_retrieval.py)、[VQA 训练入口](../BLIP/train_vqa.py)。先学第 1–3 章的形状记号；这里的 `image_embeds` 均指 ViT 的**完整**图像 token，除非写明 `[:,0,:]`。

## 1. `blip.py` 的导入和公共函数

`warnings.filterwarnings('ignore')` 关闭 Python 警告显示；对初学者而言，调试时关闭警告反而可能隐藏线索。`VisionTransformer` 是本仓库视觉类，`interpolate_pos_embed` 在图片尺寸变化时重采样位置向量。`BertConfig/BertModel/BertLMHeadModel` 来自本仓库 `med.py`，`BertTokenizer` 来自 Transformers。`torch` 做张量运算，`nn` 建层，`F` 是函数式接口；`os.path.isfile` 判断路径；`urllib.parse.urlparse` 拆 URL；`download_cached_file` 下载并缓存远程模型文件。各项只是导入工具，执行哪些由函数调用决定。

`init_tokenizer()` 的两个特殊词：`[ENC]` 用于图文编码；`[DEC]` 用于生成起点。`create_vit(vit,image_size,...)` 返回 `(visual_encoder,vision_width)`，其中第二项是整数维度。`is_url(url_or_filename)` 中 `url_or_filename` 是可能为链接、也可能为本地文件的字符串；`urlparse(...).scheme` 是协议名（如 `https`），只有 `http/https` 视为 URL。

`load_checkpoint(model,url_or_filename)` 的 `model` 是已构建好的 PyTorch 模型，`checkpoint` 是从文件读到的字典，`state_dict=checkpoint['model']` 才是“层名→参数张量”的映射。`interpolate_pos_embed` 在 224 预训练模型用于 384/480 图时调节位置向量。遍历 `model.state_dict().keys()` 时，若检查点中同名张量形状不同就删除该项，避免 `load_state_dict` 形状报错；`strict=False` 容许任务头未加载。返回 `(model,msg)`，`msg.missing_keys` 是未成功载入的层名。**`strict=False` 不表示任何错误都可忽略**，要看缺失的是新任务头还是关键主干层。

## 2. `BLIP_Base`：三种特征提取方式

`BLIP_Base` 的 `Base` 表示通用基础特征模型。构造器建 `visual_encoder`、`tokenizer`、`text_encoder`。`forward(image,caption,mode)` 中 `mode` 必须为 `'image'`、`'text'`、`'multimodal'`。源码先 tokenizer 文字，即使最终 mode 为 `'image'` 也会分词，这是当前实现的执行顺序。

| `mode` | 源码主要步骤 | 返回值 |
| --- | --- | --- |
| `'image'` | `visual_encoder(image)` | 全部图像 token `[B,P,Dv]` |
| `'text'` | `text_encoder(input_ids,...,mode='text')` | 全部文字 token `[B,T,Dt]`，不读图 |
| `'multimodal'` | ViT 编图；首 token 改 `[ENC]`；文字跨注意力读图 | 融合后的全部文字 token `[B,T,Dt]` |

`text.input_ids[:,0] = enc_token_id` 在多模态分支是**原地修改 tokenizer 输出**；此函数之后不再复用未改的 ID，所以与预训练代码先 `.clone()` 的情形不同。`image_atts=torch.ones(image_embeds.size()[:-1],...)` 对应每个有效图片 token 一位。`return_dict=True` 让调用端可用 `.last_hidden_state`。若你只要整图/整句代表向量，还需自己取 `[:,0,:]`；这个基础模型返回的是**所有 token**。

## 3. `BLIP_Decoder`：训练和推理为什么分开

构造器中 `self.prompt='a picture of '` 是固定前缀；`prompt_length=len(tokenizer(prompt).input_ids)-1` 是前缀在 token 序列中的长度，减 1 用来排除 tokenizer 自动添加的结束符。`BLIP_Decoder.forward(image,caption)` 是**有答案的训练路径**：

1. `image_embeds=visual_encoder(image)` 得 `[B,P,Dv]`；`image_atts` 全 1。
2. `text=tokenizer(caption,padding='longest',truncation=True,max_length=40,return_tensors='pt')`。`padding='longest'` 只补到本批最长句，不像预训练固定补到 30；但超长句最多 40 token。
3. 首 token 改 `[DEC]`；`masked_fill(...pad...,-100)` 使 PAD 不算语言模型 loss；`decoder_targets[:,:prompt_length]=-100` 使固定提示词也不贡献 loss。
4. `text_decoder(...,encoder_hidden_states=image_embeds,labels=decoder_targets)` 预测下一词，返回 `.loss`；外层训练脚本对这个标量 `backward()`。

`generate(image,sample=False,num_beams=3,max_length=30,min_length=10,top_p=0.9,repetition_penalty=1.0)` 是**没有标准 caption 的推理路径**。`sample=False` 时采用 Beam Search（束搜索，保留多个候选续写），`image_embeds.repeat_interleave(num_beams,dim=0)` 把每张图的特征复制给每条束。`sample=True` 时核采样，`top_p` 控制候选词的累积概率覆盖范围，输出会随机变化。`model_kwargs` 字典中的图像 token/mask 由 `**model_kwargs` 展开成命名参数交给 `generate()`；`**` 不是乘法。

提示词先被 tokenizer 转 ID，首位改 `[DEC]`，再用 `input_ids[:,:-1]` 去掉末尾结束符，给生成留出继续写的位置。输出 `outputs` 还是整数 token ID；`tokenizer.decode(output,skip_special_tokens=True)` 转为字符串，再用 `caption[len(self.prompt):]` 去掉固定前缀。**代码里的一个细节：采样分支把 `repetition_penalty` 硬编码为 `1.1`，不使用同名函数参数；束搜索分支才传入参数值。**

`blip_decoder(pretrained='',**kwargs)` 是工厂函数，不是类；`**kwargs` 收集可变数量的命名参数并转交给 `BLIP_Decoder`。`pretrained` 非空时再调用 `load_checkpoint`。训练脚本 [`train_caption.py`](../BLIP/train_caption.py) 的一轮训练就是 `model.train()`、每批 `image.to(device)`、`loss=model(image,caption)`、`optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`。评估函数加 `@torch.no_grad()`、`model.eval()`，调用 `model.generate()`，把每张图片 ID 与句子配成结果字典。

## 4. `BLIP_Retrieval`：为何多一个 `idx`

图文检索要在很多候选中找对应项。构造结构与预训练的 ITA+ITM 相似，没有 LM 生成损失；`forward(image,caption,alpha,idx)` 返回 `loss_ita,loss_itm`。`idx` 是**图片或配对的标识序号**，不是 tensor 的第几个位置。检索数据可能一张图片有多段正确描述，所以不能只假设“本批对角线为唯一正例”。

`idx=idx.view(-1,1)` 把 `[B]` 变 `[B,1]`；`idx_all=torch.cat([idx.t(),idx_queue],dim=1)` 形成 `[1,B+Q]` 候选 ID。`torch.eq(idx,idx_all)` 利用广播得到 `[B,B+Q]`：第 i 个查询与第 j 个候选 ID 相同时为真。`pos_idx/pos_idx.sum(1,keepdim=True)` 让一行里**多个正确候选均分目标概率**；`keepdim=True` 保留 `[B,1]` 以便广播除法。`idx_queue` 与图文特征队列一起更新，保证队列里的特征与原图片 ID 对应。

ITM 负例分两条路径：`negative_all_rank=False` 只在本 GPU 批中抽；`True` 跨全部 GPU 抽。两者都先用 `mask=torch.eq(idx,idx.t())` 或跨 GPU 的 `idxs.t()` 排除**所有同 ID 真配对**，再按相似度 `torch.multinomial` 抽困难负例。`concat_all_gather()` 收集特征但不传梯度；`all_gather_with_grad()` 使用自定义 `torch.autograd.Function` 在跨 GPU 收集图片完整 token 时支持反向传播。`GatherLayer.forward(ctx,x)` 中 `ctx` 是自动求导上下文；`backward(ctx,*grads)` 定义梯度怎样回到各进程。只读单 GPU 路径时先不钻这个分布式细节。

检索评估在 [`train_retrieval.py`](../BLIP/train_retrieval.py)：先用整体图文向量计算大范围候选相似度，再对高分候选用 ITM 融合模型重新排序；最终用图找文、文找图的排名评估。`scores_i2t` 表示每张图对各文字的得分矩阵，`scores_t2i` 方向相反。`itm_eval()` 根据正确图片/文字 ID 算召回指标。读评估代码时，先区分“快速全局相似度”和“候选逐对融合重排”。

## 5. `BLIP_VQA`：问题先看图，答案再看问题

VQA = Visual Question Answering，视觉问答。`BLIP_VQA.forward(image,question,answer=None,n=None,weights=None,train=True,inference='rank',k_test=128)`：

| 参数 | 是什么 | 何时用 |
| --- | --- | --- |
| `image` | 本批图像张量 | 训练/推理都要 |
| `question` | 问句字符串列表，进入函数后被重新赋值为 tokenizer 输出对象 | 训练/推理都要 |
| `answer` | 训练时可接受答案，或推理排名时的候选答案 | 对应分支 |
| `n` | 每道问题对应多少个训练答案的计数列表 | `train=True` |
| `weights` | 各训练答案的人工标注权重 | `train=True` |
| `inference` | `'generate'` 逐词生成，`'rank'` 对候选答案排序 | `train=False` |
| `k_test` | 排名时先筛多少个候选答案 | `'rank'` |

共同前半段先 `visual_encoder(image)`，tokenizer 问句，再把问句首 token 改 `[ENC]`。训练分支中 `question_output=text_encoder(question...,encoder_hidden_states=image_embeds)` 是**问题看图后的融合文字向量**。这里的 `question_states` 名字虽含 question，实际已融合图片。一个问题可有多个候选正确答案，循环 `for b,n in enumerate(n)` 把该问题的融合状态和 mask 重复 n 次，与答案条数一一对齐；其中循环变量 `n` 会覆盖外层参数名 `n` 在循环体里的引用，是源码较容易混淆的写法。

答案 `answer` 在 tokenizer 后由字符串列表变成输出对象，首 token 改 `[DEC]`；`answer_targets` 把 PAD 变 -100。`text_decoder(...,encoder_hidden_states=question_states,labels=answer_targets,reduction='none')` 按**每个答案**返回损失，而不是立刻平均；`weights * answer_output.loss` 对答案加权；`sum()/image.size(0)` 再按问题数平均。与图像描述的解码器不同，**答案解码器看到的是“已看图的问题表示”**，所以完成了图片和问题双重条件。

推理 `inference='generate'` 时，对每个已融合的问题做 beam search，并 `decode()` 回字符串。`inference='rank'` 时，`rank_answer()` 两阶段筛选：先看答案第一个 token 的概率，在候选中选 `topk(k)`；再把问题状态复制 k 份，用解码器计算这些完整候选答案的逐词损失；负损失越高，整句条件概率越好，`argmax(dim=1)` 选最终答案编号。`tile(x,dim,n_tile)` 是仓库自己的复制并重排辅助函数，让问题状态顺序与候选答案顺序对齐；它用 `numpy.arange` 造索引，再 `torch.index_select` 按索引重排。

## 6. 本章主要 API：函数签名、输入、输出

| API | 常见输入 | 返回/副作用 | 读代码时的重点 |
| --- | --- | --- | --- |
| `torch.ones(shape,dtype=...)` | 形状元组 | 全 1 张量 | 图像 mask 并非计算出的概率 |
| `torch.cat(list,dim=d)` | 一组除 d 维外相同的张量 | 沿已有维拼接 | `dim=0` 增 batch 数，`dim=1` 增候选/token 数 |
| `torch.stack(list,dim=d)` | 同形张量列表 | 新增一维 | 与 `cat` 不同，需要每项完全同形 |
| `tensor.clone()` | 张量 | 数值副本 | 修改副本不覆盖原 token ID |
| `tensor.detach()` | 张量 | 不再向此前计算图反传的张量 | 教师/队列目标常用 |
| `repeat_interleave(n,dim=0)` | 批量张量 | 每条连续复制 n 次 | Beam Search 的图像/问题条件 |
| `F.softmax(logits,dim=-1)` | 任意实数分数 | 指定维度概率和 1 | 转成抽样权重或概率 |
| `F.cross_entropy(logits,labels)` | `[B,类别数]` 与 `[B]` 整数类别 | 标量分类损失 | 传原始 logits，不需先 softmax |
| `torch.multinomial(weights,1)` | 非负权重 | 按权重抽下标 | 权重全 0 会失败 |
| `tensor.topk(k,dim=...)` | 分数张量 | `(最高 k 个值,对应索引)` | VQA 先筛候选 |
| `tokenizer.decode(ids,skip_special_tokens=True)` | 整数 ID 序列 | 可读字符串 | 跳过 `[DEC]`、`[SEP]`、`[PAD]` |
| `model.train()` / `.eval()` | 模型 | 修改运行模式 | 决定 Dropout 等训练/评估行为，不自动计算 loss |
| `loss.backward()` | 标量 loss | 给可训练参数累积梯度 | 与 `optimizer.step()` 分开 |
| `optimizer.zero_grad()` / `.step()` | 优化器 | 清旧梯度 / 更新参数 | 顺序是清、算、反传、更新 |

`nn.Module`、`nn.Parameter`、`nn.Linear`、`nn.Embedding`、`nn.LayerNorm`、`nn.Dropout` 的具体作用见第 1、2、3 章。更长的字典式查找见[现有术语与库函数详解](04-BLIP-术语与库函数详解.md)。

## 7. 读完后自己检查的五个问题

1. 为什么 ITA 用 `image_embeds[:,0,:]`，ITM 却传完整 `image_embeds`？
2. `mode='text'` 与传 `encoder_hidden_states=image_embeds` 的默认模式分别运行哪些注意力？
3. `-100` 是注意力 mask 还是语言模型标签？
4. 检索为什么用 `idx` 标记多个正例，而预训练常只把对角线标为正例？
5. VQA 的答案解码器看的是原图 token，还是已经看图的问题状态？

能用源码中具体变量回答这五题，就已经掌握 BLIP 的主要数据流。接下来再看 [训练入口逐行详注](BLIP-pretrain-零基础逐行详注.py)，把每个 batch 的优化器步骤接上。
