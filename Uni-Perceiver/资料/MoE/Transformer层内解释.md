# Uni-Perceiver-MoE Transformer 层内解释

## 1. 阅读目标和文件

本节接着 [属性与门控解释](属性与门控代码解释.md)，回答：专家被选中以后，如何参与一次 Transformer 计算？源码为 [transformer_encoder_moe_layer.py](../../uniperceiver/modeling/layers/transformer_encoder_moe_layer.py)，包含 `MoETransformerEncoderLayer` 与 `MoEAttentionBlock`。本次新增 149 处中文注释，[核对记录](层内注释核对.json) 证明可执行 AST 与注释前、Git HEAD 一致。

Transformer 的两个主分支是自注意力 SA（Self-Attention）和前馈 FFN（Feed-Forward Network）。MoE=Mixture of Experts，混合专家，给部分线性映射准备多个专家权重，再按属性条件选择/融合。它不是把整个模型按任务拆成互不相干的模型，也不是把 softmax 的每个概率格变成一个专家。

本文件中的专家位置：QKV 线性投影、注意力输出投影、FFN 第一线性、FFN 第二线性。哪些层启用专家由上游配置决定；仅阅读本文件不能知道完整模型每层都用了专家。

## 2. 参数、变量与形状字典

| 名字 | 命名来源 | 含义 |
|---|---|---|
| `d_model / hidden_size` | dimension of model，模型维度 | 隐藏宽 $D$ |
| `nhead / num_attention_heads` | number of heads，头数 | 注意力头数 $h$，与专家数 $E$ 不同 |
| `attention_head_size` | 每个注意力头的宽度 | $d_h=D/h$，需整除 |
| `dim_feedforward` | 前馈内部维度 | FFN 第一层输出宽 $F$ |
| `src / x` | source / 局部计算变量 | 当前输入 $[B,L,D]$ |
| `history_states / kv` | 历史隐藏状态 / key-value 来源 | 投影前隐藏序列 $[B,S,D]$，并非已投影缓存 |
| `mixed_query_layer` | 未分头的 query | $[B,L,D]$ |
| `query_layer` | 分头后的 query | $[B,h,L,d_h]$ |
| `attention_scores` | 注意力分数 | $[B,h,L,S]$，softmax 前 |
| `attention_probs` | probabilities，注意力权重 | 同形状，沿 key 轴归一化 |
| `context_layer` | 汇总内容的上下文 | 合并头后 $[B,L,D]$ |
| `gate_decision` | 门控决策 | 专家索引与权重，可复用 |
| `cfg` | configuration，配置 | 层级属性对象，不能直接用普通 dict 替代点访问 |

图像与文字一般已经被上游嵌入为相同隐藏宽度的 token 序列。本层不是把原 RGB 像素转换成 patch，也不进行文本分词。

注意旧英文 docstring 沿用 PyTorch 标准层模板，提到 `[seq,batch,feature]`、fast path、NestedTensor；当前实际自定义注意力按 `[B,L,D]` 解包，没有那套快速路径实现。`self.batch_first` 保存一个标志，不转置数据。构造中仅主进程把该参数改 False，属于原实现边界，不能照模板传 `[L,B,D]`。

## 3. import 和构造接口

- `torch` 是 PyTorch；`nn` 是 neural network 模块，提供注册子模块/参数、线性、归一化与 Dropout；`F` 是 functional 函数式接口，`F.linear` 用显式权重、`F.softmax` 转权重。
- `typing.Optional[T]` 表示 T 或 None，`Union` 表示允许多类型，`Callable` 表示可调用对象。这里只提供类型提示，不验证形状。`Any`、`get_act_layer` 虽导入但本文件未使用。
- `functools.partial` 是标准库偏函数：先绑定专家公共配置，后面分别传模板线性层。它创建可调用构造器，不会立刻执行模型前向。
- `timm.DropPath` 是随机深度；`nn.Dropout` 是逐元素随机置零，两者位置和作用不同。
- 项目的 `LayerNorm/FP16LayerNorm` 是可选归一化实现；`DeepPrompt` 提供各层学习的提示 token；`comm` 是进程/通信工具；`TaskMoE` 实施门控与专家线性计算。
- `math.sqrt` 用于注意力缩放；`autocast(enabled=False)` 关闭自动混合精度，但不关闭梯度，原代码再显式 `.half()`。

构造的 `activation` 可为字符串或函数，字符串通过 `get_activation` 解析。`layer_norm_eps` 是归一化分母的数值稳定小常数，与门控噪声和 label smoothing 无关。`device/dtype` 虽在接口中出现，但 `factory_kwargs={}` 没有使用它们；不能靠传此参数就认为所有层自动放到指定设备。

`ffn_moe/attn_moe` 是局部开关，`cfg.MOE.MOE` 是全局开关。`TaskMoE` 的公共配置包括专家数 $E$、选中数 $k$、噪声和容量接口选项。当前只支持 attribute 门控类型。通用容量/Tutel/RTS 参数传入，不证明当前属性融合分支进行了逐 token 分发或跨设备通信。

## 4. 一次 Pre-LN 残差计算

【必须会】常用 Pre-LN（Layer Normalization 在子层之前）顺序：

$$X'=X+\operatorname{SA}(\operatorname{LN}_1(X)),\qquad Y=X'+\operatorname{FFN}(\operatorname{LN}_2(X')).$$

$X,X',Y$ 都是 $[B,L,D]$。LayerNorm 沿最后的特征维规范化，不把不同图片或 token 混为一个均值。残差加法保持原输入，并学习增量；注意力与 FFN 不直接覆盖全部原信息。

若有历史隐藏状态，也用相同 `norm1` 归一化它，再和当前特征拼接，确保它们进入注意力投影时位于相同处理阶段。

分支输出先用逐元素 Dropout，再用 DropPath（按样本随机去掉该残差分支），最后可乘 LayerScale 的通道参数 $\gamma_1/\gamma_2\in\mathbb R^D$。它们初值 `ls_init_values` 较小，令初始层输出更接近输入。外层 `x+...` 才完成残差加法。`.eval()` 关闭这两个随机正则模块；仍不自动关闭梯度。

Post-LN 的理论顺序是 $\operatorname{LN}(X+\text{子层}(X))$。当前源码 FFN 分支写为 `self.norm2(x+self._ff_block(x),**kwargs)`，并没有把 kwargs 传给 `_ff_block`，反而传给 norm2；有属性参数时可能报错。理论公式与当前可运行配置必须区分。

## 5. 注意力：专家参数到多头运算

### 5.1 QKV 合并线性与专家融合

【必须会】一个合并线性层权重形状 $[3D,D]$。输入 $[B,L,D]$ 映射为 $[B,L,3D]$，最后维按顺序切成 Q、K、V，每段 $D$。

PyTorch Linear 的权重按 `[out_features,in_features]` 保存：$f(x)=xW^\top+b$。属性选中两个线性专家时，融合：

$$W_*=g_1W_1+g_2W_2,\quad b_*=g_1b_1+g_2b_2,\quad f_*(x)=xW_*^\top+b_*.$$

在该组门控权重一致的条件下，等价于 $g_1f_1(x)+g_2f_2(x)$，可减少重复线性计算。只是同一线性投影可融合，不能跨过 softmax、GELU/ReLU 等非线性后还声称整个模块可合并为一个线性层。

无历史时 QKV 一次计算，并返回 `gate_decision`。源码把局部 `hidden_states` 覆盖为投影结果，所以此处它从 D 维变成3D维；再 `chunk(3,dim=-1)` 拆开。输出投影 `dense` 复用 QKV 的门控，而不是对每个注意力上下文再独立决定专家。

### 5.2 分头、点积与值汇总

拆头函数先 `view(B,L,h,d_h)`，再 `permute` 成 $[B,h,L,d_h]$。Q 的长度为 $L$，K/V 的长度为 $S$。无历史且无提示时常有 $S=L$。

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\qquad C=AV.$$

$Q:[B,h,L,d_h]$，$K^\top:[B,h,d_h,S]$，得分/权重 $[B,h,L,S]$；乘 $V:[B,h,S,d_h]$ 得 $C:[B,h,L,d_h]$。随后换轴到 $[B,L,h,d_h]$，contiguous 整理内存，view 合并为 $[B,L,D]$，经输出线性投影。

为什么除 $\sqrt{d_h}$：维度增长会增大随机点积尺度，让 softmax 容易过度集中；这个缩放用于控制尺度。`SCALE_MULTI_BEFORE` 选择先缩放 Q 再点积，或点积后缩放，数学相同，浮点舍入/溢出情况可能不同。

`FORCE_SOFTMAX_FP16=True` 时原代码显式把输入 `.half()`，不是转 FP32 提升稳定性。值张量精度、设备和周围混合精度配置要配合；当前局部检查关闭它，未验证所有精度组合。

### 5.3 mask 的语义与全遮盖边界

`attn_mask:[L,S]` 标记序列结构，`key_padding_mask:[B,S]` 标记每图无效 key。两者都有时转布尔再逻辑或，经广播得 $[B,L,S]$；再加头共享轴 $[B,1,L,S]$，乘 -10000 加入得分。当前约定 1/True 表示禁止。

这个接口不是“传任意加性负无穷 mask 都行”：单独 mask 分支还会乘 -10000，两个分支又 bool 转换，必须遵守当前0/1语义。

【建议会】若整行都被遮盖，给所有格子加有限负常量，softmax 仍归一化成非零概率，行和约1；不能声称“所有key禁止，所以输出一定零”。本次实际原注意力运行证实这一点。调用方应避免全无效行或明确处理其语义，本文没有自行更改算法。

注意力概率 Dropout 在训练时置零并缩放某些连接，一次结果的行和不一定仍1；评估时 Dropout 关闭。概率热图本身也不自动构成可靠因果解释。

## 6. 历史状态和 Q/KV 切片

【建议会】本层 `history_states` 是投影前隐藏状态，不是 BEiT-3 caption 那种已经生成好的 per-layer key/value 缓存。外层先拼历史与当前，再送给注意力重新投影。

有历史时当前输入只需要 Q，完整历史/当前的来源需要 K/V：

- `mode='q'` 只取 $[3D,D]$ 权重的前 D 行，输出 $[B,L,D]$。
- `mode='kv'` 取后 2D 行，输出 $[B,S,2D]$，再切为两个 $[B,S,D]$。
- 专家路径的 KV 复用 Q 的 `gate_decision`，输出 dense 也复用这个决策。复用的是各条件组门控结果，并非强制视觉和文字每个组选择相同专家。

非专家路径用 `F.linear` 手工切同一 QKV 权重的对应行。参数存储始终还是一个合并层，并没有注册三个新的线性层。

源码在历史拼接后留有“修改 mask”的 TODO。调用方要提供覆盖扩展后 key 长度的 mask，不能把原 $L\times L$ 无条件用于 $L\times S$。本次局部检查明确传完整 mask 的最后两行，因此只证明该输入设置的数学一致性。

## 7. FFN 的门控复用与非线性

FFN 原始结构：

$$\operatorname{FFN}(X)=W_2\,\phi(W_1X+b_1)+b_2,$$

这里是概念列向量写法，源码 Linear 对最后维采用行向量计算。$W_1$ 从 D 扩到 F，激活 $\phi$ 为 ReLU/GELU 等，再投影 F→D。

开启 FFN MoE 时，第一专家层返回结果与门控；`FFN_SHARE_GATE_DECISION=True` 让第二专家层复用，False 则传 None 触发第二模块重新计算。即使共享门控，第一/第二线性专家参数不同，激活也仍在两者之间。第一层融合后激活、再第二层融合，通常不等价于对各个完整专家 MLP 输出加权求和。

属性门控也不同于对每个 token 激活任意选择路由。本版本 `FusedExperts` 按一组/两组属性沿序列分段，组边界取 `sample_info['data_cum_length'][1]`。一组时整个序列同路由，两组时分别处理两段。详见先前属性说明，论文8位属性与本地具体编码有已记录差异，不能在本节重新定义掩盖它们。

## 8. DeepPrompt 与分段元数据

DeepPrompt 生成 $[B,P,D]$，放在 key/value 序列前部，当前 query 不增加提示位置，所以输出还是 $[B,L,D]$。若 Pre-LN，提示也做 norm1。

新增提示 key 补 P 列0到序列mask、补 P 项0到padding mask，允许query读提示。代码还原地写 `kwargs['sample_info']['pe_length']=P`。`**kwargs` 只复制外层映射，嵌套 sample_info 仍可能由调用者共享，因此这项变更有状态影响。

在 `FusedExperts.top2_expert_forward` 的 `mode='kv'` 两组分段里，第一段边界加 pe_length，使前置提示归入第一段；不能只看图像/文字原始 token 数切分。当前局部检查关闭 DeepPrompt，并未运行这整条提示路径。

## 9. 实际核对、配置边界与未完成事项

[检查脚本](../../复现/scripts/检查MoE注意力层.py) 每行有注释，通过 AST 提取原两个类，真实运行稠密注意力；门控接口使用 `TraceTaskMoE` 替身，仅验证传参顺序/复用。实际结果见 [输出](../../复现/outputs/MoE注意力层检查.json)。

1. $D=4,h=2,B=1,L=4$，原注意力与独立公式手算最大误差0。
2. 当前query取最后2项、历史key/value源取完整4项，结果等于完整注意力最后两项，误差0。
3. 全行mask时概率仍非零、行和约1，证实有限负值边界。
4. Q、KV、dense 三处复用同一门控对象，使用接口替身证实；不证明真实多专家数值/梯度。
5. 全局 MoE 关闭时构造实际抛 `UnboundLocalError`，因为 `MoE_layer` 未定义却被引用。
6. Post-LN 稠密 FFN 传属性 kwargs 时实际抛 `TypeError`，额外键被交给普通 LayerNorm。

原版本判断还会把 PyTorch 2.6 当旧接口；此文件该标志未实际决定自定义注意力计算，本次运行不受它影响。独立 Q/K/V 构造分支存在，但 forward 明确要求 unify QKV，不能据构造分支宣称该模式支持。

【必须会】四个专家线性位置、Pre-LN残差、Q/K/V形状、门控复用、组内参数融合条件。第二遍理解历史隐藏重投影、提示分段、mask广播、概率Dropout与LayerScale。融合内核/分布式专家通信/混合精度组合可暂缓。

| 论文方法 | 本地代码对象 | 证据边界 |
|---|---|---|
| 属性式门控与线性融合 | TaskMoE / FusedExperts | 先前有局部运行，非完整训练 |
| SA MoE | MoEAttentionBlock.qkv_proj/dense | 详细注释；稠密数学真实局部运行，专家传参用替身 |
| FFN MoE | `_ff_block.linear1/linear2` | 详细注释，未完整专家前向 |
| Transformer 残差/归一化 | MoETransformerEncoderLayer.forward | 静态分析与两个配置异常实测 |
| 多任务成绩与干扰消除 | 完整训练/评估/梯度分析 | 本次未复现 |

整体状态仍是未复现完整 Uni-Perceiver-MoE；本次完成层内源码深读和有边界的局部运行。待继续的是完整真实专家前向、剩余数据调用链、论文表5/附录与梯度干扰公式、环境兼容、翻译审校。
