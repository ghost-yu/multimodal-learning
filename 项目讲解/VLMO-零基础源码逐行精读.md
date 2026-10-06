# VLMO：逐步读懂共享注意力与模态专家

对应 [VLMO 概览](06-VLMo.md)、[多路 Transformer](../UniLM/vlmo/vlmo/modules/multiway_transformer.py)、[VLMo 主模块](../UniLM/vlmo/vlmo/modules/vlmo_module.py)、[损失目标](../UniLM/vlmo/vlmo/modules/objectives.py)。这里讲本地源码的核心调用链；模型还有 checkpoint 转换、位置偏置插值、完整检索评估等工程支线，第一次可沿函数名以后再读。

## 0. 先用一层的结构理解 MoME

MoME = **Mixture of Modality Experts**，模态专家混合。每层先用一套共享自注意力让 token 交换信息，随后按输入模态选择不同前馈网络 FFN。`image` 用图像 FFN，`text` 用文字 FFN，`vl`（vision-language，图文）可以用图文 FFN；若当前层未创建图文 FFN，则把混合序列中的文字和图像段拆开，各走对应专家。

这不是随机挑一个专家的门控 MoE。此源码中的选择主要由明确的 `modality_type` 字符串和层配置决定。`self.attn` 在同一 Block 内被图像、文字、图文路径共用；`self.mlp_text`、`self.mlp_imag`、可选 `self.mlp_vl` 是参数不同的网络。

下文 `B` 批量数、`N` token 总数、`T` 固定文字长度、`P` 图像 token 数、`D` 特征宽度、`H` 注意力头数、`d=D/H` 每头宽度。

## 1. 导入与库函数地图

| 名字 | 来源、命名含义 | 常用调用在本源码的作用 |
|---|---|---|
| `torch` | PyTorch 张量库 | `torch.cat` 拼 token；`torch.linspace` 建逐层递增概率；`torch.ones` 建 mask |
| `torch.nn as nn` | neural network 可训练层 | `nn.Linear`、`nn.LayerNorm`、`nn.ModuleList`、`nn.Parameter` |
| `torch.nn.functional as F` | functional 无状态函数 | `F.linear` 一次算 Q/K/V；`F.cross_entropy` 算分类损失 |
| `functools.partial` | 标准库预填参数 | `partial(nn.LayerNorm,eps=1e-6)` 给 LayerNorm 设置默认 eps |
| `DropPath`（timm） | stochastic depth 随机深度 | 训练时按概率跳过一个残差分支；与逐元素 `Dropout` 不同 |
| `to_2tuple`（timm） | 尺寸参数规范化 | 使 `224` 变 `(224,224)`，便于处理高与宽 |
| `trunc_normal_`（timm） | 截断正态初始化 | 原地初始化 CLS、位置向量和层权重 |
| `register_model`（timm） | 模型注册装饰器 | 令 `timm.create_model` 可按名字创建 `vlmo_base_patch16` |
| `rank_zero_info`（Lightning） | 只在主进程打印日志 | 避免多 GPU 重复输出 |
| `pytorch_lightning as pl` | 训练框架 | `LightningModule` 的 `training_step` 等回调 |
| `torch.distributed as dist` | 多进程通信 | `get_world_size`、`all_gather` 扩大对比学习负例池 |

`nn.ModuleList([...])` 与普通 Python 列表的区别：它把每个 Block 正式登记为模型子模块，这样优化器、`.to(device)`、`state_dict()` 才会处理其中参数。`nn.Parameter` 把一个张量登记为可训练参数；`register_buffer` 则登记需保存/迁移但不训练的张量。末尾 `_` 的函数一般原地修改，例如 `masked_fill_`。

## 2. `Mlp`：最简单的专家

位置：[multiway_transformer.py 的 `Mlp`](../UniLM/vlmo/vlmo/modules/multiway_transformer.py)。`MLP` 是 multilayer perceptron，多层感知机；这里实质是两层线性层和一次 GELU 激活。`in_features=D` 是输入宽度；`hidden_features` 是中间宽度，通常 `int(D*mlp_ratio)`；`out_features` 默认回到 D，才能与原输入做残差相加。

```text
x [B,N,D] → fc1 [B,N,hidden] → GELU → Dropout
          → fc2 [B,N,D]      → Dropout → 返回
```

`nn.Linear` 会把所有前导维保持不变，只变最后一维，所以不必写循环逐个 token 处理。`GELU` 给网络非线性；没有它，两层线性层可以合并成一层线性变换。`Dropout` 仅在训练模式随机置零部分激活，评估模式不随机丢弃。`x` 是各行不断更新的当前特征，切勿认为始终是入口值。

## 3. `Attention.forward`：逐行拆 Q、K、V

位置：[multiway_transformer.py 的 `Attention.forward`](../UniLM/vlmo/vlmo/modules/multiway_transformer.py)。`B,N,C=x.shape` 是 Python 解包，`C` 指本层 feature channels，等于前述 D。字母 Q/K/V 分别是 query（我想找什么）、key（我能提供什么）、value（实际取出的内容）。

1. `qkv_bias=None`：默认不用偏置。如果有 `self.q_bias`，`torch.cat((q_bias, zeros_like(v_bias), v_bias))` 合出长度 `3D` 的偏置；中间的 K 偏置固定为零。`zeros_like` 复制形状和设备，不复制数值。
2. `F.linear(input=x,weight=self.qkv.weight,bias=qkv_bias)` 一次把 D 映射到 3D，结果 `[B,N,3D]`。与建立三次 `nn.Linear` 的功能相似，但共享一次大矩阵乘。
3. `.reshape(B,N,3,self.num_heads,-1)` 划分为 Q/K/V、H 个头、每头 d；`-1` 让 PyTorch 算 d。`.permute(2,0,3,1,4)` 调整为 `[3,B,H,N,d]`，便于 `qkv[0]`、`[1]`、`[2]` 分别取 Q/K/V。
4. `q=q*self.scale`，其中常见 `scale=d**-0.5`，避免点积数值随维度变大。`q.float() @ k.float().transpose(-2,-1)` 产生 `[B,H,N,N]` 的每对 token 分数；`.float()` 临时提升到 float32 以增加计算稳定性。
5. 若有 `relative_position_bias`，`.unsqueeze(0)` 增一个 batch 轴，再广播加到每头分数上。它表达两个 token 相对位置，不等于输入 token 自身的绝对位置 embedding。
6. `mask.bool()` 将有效位转布尔。`~mask[:,None,None,:]` 的形状为 `[B,1,1,N]`：`None` 在索引里插入长度为 1 的轴。`masked_fill(...,-inf)` 把无效 key 的分数设为负无穷，softmax 后其概率为零。
7. `attn.softmax(dim=-1)` 在“被关注的 N 个 key”方向归一化。`.type_as(x)` 把输出转回入口精度。`attn @ v` 做加权求和，接着 `.transpose(1,2).reshape(B,N,C)` 把 H 个头拼回 D。最后 `self.proj` 做线性投影，`self.proj_drop` 做 dropout。

注意 `mask` 阻止其他位置**读取无效 key**；代码没有在这里把无效 query 行直接删除。`attn` 变量先表示未归一化分数，后又被赋值为概率，最后返回的却是输出 `x`，不返回注意力矩阵。读代码时看每次赋值之后的新含义。

## 4. `Block.forward`：共享与专家到底在哪行

```python
x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask,
                                                relative_position_bias=relative_position_bias))
```

最里面 `norm1(x)` 做 LayerNorm；`attn(...)` 是**同一个**共享注意力对象；`gamma_1` 是 LayerScale，可训练的逐特征缩放；`drop_path` 在训练时可能跳过整条残差分支；最后与原 `x` 相加保留信息。此行执行完，`x` 已是结合新注意力结果的张量。`gamma_2` 对后面的 FFN 分支做类似缩放。

```python
if modality_type == 'image':
    x = x + drop_path(gamma_2 * mlp_imag(norm2_imag(x)))
elif modality_type == 'text':
    x = x + drop_path(gamma_2 * mlp_text(norm2_text(x)))
else:
    # 图文混合：有 mlp_vl 时整条序列走它；没有则按固定文字长度切开
```

以上是简写帮助阅读，完整精确语句在源码。`norm2_imag`、`norm2_text`、`norm2_vl` 也各有独立参数；“专家”不仅是 MLP 权重不同。没有 `mlp_vl` 时，`x_text=x[:,:self.max_text_len]`、`x_imag=x[:,self.max_text_len:]`；分别经两个 FFN，`torch.cat([x_text,x_imag],dim=1)` 按原顺序接回。这里**要求输入混合序列前段的文字 token 数与 `max_text_len` 的布局约定一致**。图文 token 在共享注意力阶段已交互，拆开走 FFN 并不抹掉已有的信息。

`with_vlffn` 由 `i >= vlffn_start_layer_index` 决定，所以靠后层可有专用图文 FFN；在 `textmlm` 纯文字阶段，代码把起始层设为 `depth`，相当于该阶段不创建图文 FFN。`modality_type='vl'` 不是一种张量类型，只是 Python 字符串，供 `if/elif/else` 选路径。

## 5. 图片变 token 与 `MultiWayTransformer`

`PatchEmbed` 用核大小与步长等于 patch 大小的卷积，把 `[B,3,H,W]` 变 `[B,D,H/p,W/p]`。`visual_embed(_x)` 随后 `flatten(2)` 把二维网格展平，`transpose(1,2)` 成 `[B,P,D]`；`self.cls_token.expand(B,-1,-1)` 把同一个可训练 CLS 参数显示为 B 行，`-1` 意味该维不变。`torch.cat((cls_tokens,x),dim=1)` 把 CLS 放最前。可选的 `pos_embed` 是绝对位置向量；`relative_position_bias` 则在每层注意力分数里使用。`x_mask=torch.ones(B,P+1)` 因没有图像 padding，所有位置有效。

构造器里 `num_patches=self.patch_embed.num_patches` 由图像尺寸和 patch 大小计算。`dpr=[x.item() for x in torch.linspace(0,drop_path_rate,depth)]` 给每层逐渐增大的 DropPath 概率；`.item()` 从零维张量取 Python 数字。`nn.ModuleList([Block(...) for i in range(depth)])` 建 depth 层，每层参数独立。`self.norm` 在所有层后做末归一化。`no_weight_decay()` 返回不需权重衰减的参数名字集合，供优化器构建逻辑查询。

## 6. 从 batch 到三种推理函数

在 [VLMo.infer](../UniLM/vlmo/vlmo/modules/vlmo_module.py)：`batch` 是数据整理后的字典，`text_ids` 是 `[B,T]` 整数，`text_masks` 表示有效字词；`img=batch[imgkey][0]` 从图片视图列表取张量。`text_embeddings(text_ids)` 得 `[B,T,D]`，`visual_embed(img)` 得 `[B,P,D]`；分别加类型 embedding，再 `torch.cat(...,dim=1)` 得图文联合序列 `co_embeds [B,T+P,D]`。`co_masks` 也按同序拼。每层 `blk(x,mask=co_masks,modality_type='vl',relative_position_bias=...)` 执行共享注意力与对应专家。最终从 `x` 前 T 位置取 `text_feats`、后 P 位置取 `image_feats`，`self.pooler(x)` 给整对图文 `cls_feats`。

`infer_text` 只编码文字并走 `modality_type='text'` 路线；`infer_image` 只编码图片并走 `modality_type='image'` 路线。它们输出的 `cls_feats` 可做 CLIP 风格的 ITC。`infer` 的融合 `cls_feats` 则适合 ITM/VQA 等需要细粒度交互的任务。`infer_text_mlm` 是纯文字遮词阶段路径；不要把“同名 `infer`”理解为只能在推理时使用，这些函数训练时一样会调用。

## 7. ITC、ITM、MLM：为什么顺序重要

在 [VLMo.forward](../UniLM/vlmo/vlmo/modules/vlmo_module.py)，`self.current_tasks` 是本轮要做的任务名列表；`ret.update(objectives.compute_itc(...))` 把返回字典并入 `ret`。当同时启用 ITC 与 ITM，先算 ITC：因为 `compute_itm_hardneg` 需要 `ret['itc_i2t_logits']` 与 `ret['itc_t2i_logits']` 用作难负例采样概率。如果只配置 `itm` 而未产生这些键，原实现会 `KeyError`。

`compute_itc` 先调用 `infer_image` 和 `infer_text` 各自编码，取单位向量 `image_features/text_features [B,D]`，`logit_scale.exp().mean()` 将对数尺度变正。多卡时 `dist.all_gather` 收集其他卡向量以增加负例。矩阵 `logits_per_image=scale*all_image_features@all_text_features.t()`，其对角线是正确配对；`ground_truth=torch.arange(总样本数)` 给每行的正确列。图找文和文找图各用 `F.cross_entropy` 算一次，再平均。源码还针对 `cls_vlffn_feats` 计算第二套对比损失，最后两套再平均。这是读 VLMO 与简单 CLIP 实现时的重要区别。

`compute_itm_hardneg` 根据上述相似度选错图/错文，调用融合 `infer` 对真图文与两类错配分别编码，得到三组 CLS；`itm_score` 输出真假分数，用交叉熵训练。`compute_mlm` 用遮词版 `text_ids_mlm` 与图片共同编码，再由 `mlm_score` 对每个文字位置预测词表，`ignore_index=-100` 跳过无须预测的位置。训练钩子 `training_step` 以 `vlmo_utils.set_task(self)` 设任务，调用 `self(batch)`，然后把返回字典中键名含 `loss` 的值加总交给 Lightning 反传。

### 小型复现的真实边界

`compute_itc(...,aggregate=True)` 默认会调用 `dist.get_world_size()`，依赖初始化的分布式进程组。原文件中 `aggregate=False` 分支没有创建后续还要读取的 `logits_per_vlffn_image`，直接改成 False 并不是完整的单卡解决方案。学习时可先追模型前向与形状；要跑单卡小实验，需要对这一分支作明确补全或按项目训练入口初始化分布式环境。

## 8. 隔很远再次出现的名字

| 名字 | 初始来源 | 后面读到时应回想 |
|---|---|---|
| `self.blocks` | `MultiWayTransformer.__init__` 创建的 `ModuleList` | 每个元素是一层共享注意力＋多路 FFN |
| `blk` | `for i,blk in enumerate(self.transformer.blocks)` | 当前第 i 个 Block；不是整个 Transformer |
| `x` | 当前函数入口或 `co_embeds` | 每层都会重新赋值；只能根据当前位置确定是否已融合 |
| `text_embeds` | `self.text_embeddings(text_ids)` | 尚未进共享注意力的文字向量 |
| `text_feats` | 联合 `x` 的前 T 个位置 | 已融合图像信息，可给 MLM |
| `image_features` | `infer_image()['cls_feats']` | 全局图像单位向量，给 ITC |
| `cls_vlffn_feats` | 单模态路径再走后段图文 FFN 得到 | 第二套 ITC 对比向量，不等于首套 `cls_feats` |
| `ret` | `forward` 开头的空字典 | 各目标函数用 `update` 逐渐加入损失、logits、标签 |

自检：如果 `B=2,T=40,P=197,D=768`，联合序列为 `[2,237,768]`；共享注意力的分数矩阵对每个头是 `[2,237,237]`；后面没有专用图文 FFN 的层会按前 40 个文字位置与后 197 个图像位置拆开。
