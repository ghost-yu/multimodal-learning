# 06｜VLMo：共享注意力与模态专用专家

> 最新完整版本：[VLMo 的 pdf2zh 中文论文](../UniLM/vlmo/中文论文.pdf)与[原仓库内的论文、代码精读](../UniLM/vlmo/解释.md)。

> **从零开始读源码**：建议配合 [VLMO 零基础源码精读](VLMO-零基础源码逐行精读.md) 和 [带注释的多路 Transformer 副本](四项目教学源码/VLMO-multiway_transformer-详注.py)。其中逐步说明共享注意力、文字/图片/图文 FFN 路由、主要张量操作和训练目标。

> 源码：[UniLM/vlmo](../UniLM/vlmo)；论文：[VLMo.pdf](../papers/VLMo.pdf)。此项目与 BEiT-3 位于同一个 [UniLM](../UniLM) 仓库，下面只讲 `vlmo` 目录。

## 一、先理解“模态专家”到底是哪部分

Transformer block 常分两大步：**注意力**负责 token 之间交换信息，**FFN/MLP** 对每个 token 的特征再加工。VLMo 的关键做法是：注意力部分共享；FFN 按图像、文字、图文情况选择不同参数。不是把整个模型复制三份，也不是每次随机找一个专家。

```text
输入 token → 共享 self-attention → 根据模态选 FFN → 下一层
                               ├→ 图片 FFN
                               ├→ 文字 FFN
                               └→ 图文 FFN（配置存在时）
```

## 二、文件分工

| 文件 | 负责什么 |
|---|---|
| [README.md](../UniLM/vlmo/README.md) | 分阶段训练命令与下游任务说明 |
| [run.py](../UniLM/vlmo/run.py) | 训练/测试入口 |
| [vlmo/config.py](../UniLM/vlmo/vlmo/config.py) | 模型、任务和训练配置 |
| [vlmo/modules/multiway_transformer.py](../UniLM/vlmo/vlmo/modules/multiway_transformer.py) | 共享注意力、多路 FFN、图像 patch embedding |
| [vlmo/modules/vlmo_module.py](../UniLM/vlmo/vlmo/modules/vlmo_module.py) | 处理单模态/图文输入，路由任务 |
| [vlmo/modules/objectives.py](../UniLM/vlmo/vlmo/modules/objectives.py) | MLM、ITC、ITM、VQA 等目标 |
| [vlmo/datamodules](../UniLM/vlmo/vlmo/datamodules) | 数据整理与 batch |

## 三、从 `Block.forward()` 一行一行看

[multiway_transformer.py:184](../UniLM/vlmo/vlmo/modules/multiway_transformer.py) 的第一部分类似：

```text
x = x + gamma_1 × Attention(LayerNorm(x))
```

`LayerNorm` 先把数值尺度稳定下来；`Attention` 让 token 互相交换信息；`gamma_1` 调节新信息强度；加回原来的 `x` 是残差连接。**无论传 image、text、vl，都调用同一个 `self.attn`**。这就是共享部分。

后面代码按 `modality_type` 分支：

- `image`：`norm2_imag` 后走 `mlp_imag`；
- `text`：`norm2_text` 后走 `mlp_text`；
- `vl`：若有 `mlp_vl` 则走它，否则把文字 token 和图像 token 分段，各走对应 FFN 再拼回来。

FFN 一般由两层线性层及非线性激活组成。它处理**每个 token 的特征**，不同于注意力负责的 token 间交流。即便两类 token 经同一个注意力交流，VLMo 仍可以让图像和文字保留各自更合适的加工方式。

## 四、单模态与图文输入路径

[vlmo_module.py](../UniLM/vlmo/vlmo/modules/vlmo_module.py) 中有三类入口：

- `infer_image()`、`infer_image_ft()`：只给图像。图像经 `visual_embed()` 切 patch，再走 `modality_type='image'` 的 block。
- `infer_text()`、`infer_text_mlm()` 等：只给文本。文字 ID 经 embedding，再走 `modality_type='text'` 的 block。
- `infer()`：同时给图文。文本 token 与图像 patch 加类型 embedding 后拼接，连有效位一起拼；逐层以 `modality_type='vl'` 运行。最后按原文本长度切回 `text_feats`、`image_feats`，另取 `cls_feats`。

假设文字 32 个 token、图像 197 个 token，拼接后的主序列是 `[B,229,D]`。在混合分支里，代码用文本最大长度划分前后两段，因此这个“分界点”必须与实际组织序列的方式一致。

## 五、目标函数在哪儿

[objectives.py](../UniLM/vlmo/vlmo/modules/objectives.py) 把不同训练目标拆成函数：

- `compute_textonly_mlm()`：只给文字，遮词并预测。帮助模型学语言。
- `compute_mlm()`：给图文，利用图像与上下文预测被遮词。
- `compute_itc()`：图、文整体向量做对比，让正确配对靠近。
- `compute_itm_hardneg()`：选比较难的错误配对，用融合模型判真假。
- `compute_vqa()`、`compute_nlvr2()`：下游问答和视觉推理。

`VLMo.forward()` 根据当前任务决定调哪些函数；`run.py` 根据配置启动训练。论文中的训练分阶段，README 列出视觉、语言、图文各阶段命令。初读代码先看每一阶段**输入的数据模态**和**对应损失**，不需要同时掌握分布式与 DeepSpeed 细节。

## 六、与 ViLT 的对照

ViLT 也会把图文 token 拼在一起，但主要使用一套共享 Transformer block。VLMo 的重要新点在 [Block](../UniLM/vlmo/vlmo/modules/multiway_transformer.py) 内部：注意力共享，FFN 依据输入模态切换。读懂这个差异就抓住论文核心，不必被大量任务配置淹没。

**本项目完成标准**：能在 `Block.forward()` 指出共享注意力、图像 FFN、文本 FFN、图文 FFN；能解释 `infer_image()` 与 `infer()` 为什么传入不同 `modality_type`。

## 七、给第一次读大模型源码的人：从哪一行开始

原仓库有训练、数据转换、分布式评估等许多辅助文件。现在以下**主执行链文件已直接补中文行注释**，可以依次打开，不必从文件夹第一页顺读：

| 顺序 | 文件 | 你要回答的问题 |
|---|---|---|
| 1 | [config.py](../UniLM/vlmo/vlmo/config.py) | 这次训练用什么数据、什么模型、哪些损失？ |
| 2 | [run.py](../UniLM/vlmo/run.py) | 配置怎样变成 DataModule、VLMo 模型和 Trainer？ |
| 3 | [multitask_datamodule.py](../UniLM/vlmo/vlmo/datamodules/multitask_datamodule.py) | COCO/VG/SBU/GCC 怎样拼为一个 DataLoader？ |
| 4 | [base_dataset.py](../UniLM/vlmo/vlmo/datasets/base_dataset.py) | Arrow 中的图片、文字如何转成 batch？ |
| 5 | [multiway_transformer.py](../UniLM/vlmo/vlmo/modules/multiway_transformer.py) | patch、Q/K/V、共享注意力和三个 FFN 在哪里？ |
| 6 | [vlmo_module.py](../UniLM/vlmo/vlmo/modules/vlmo_module.py) | 文字、图像、图文各自走什么路径？ |
| 7 | [objectives.py](../UniLM/vlmo/vlmo/modules/objectives.py) | MLM、ITC、ITM 等损失怎么计算？ |
| 8 | [heads.py](../UniLM/vlmo/vlmo/modules/heads.py) | CLS 向量最后怎样变成分类分数或词预测？ |
| 9 | [vlmo_utils.py](../UniLM/vlmo/vlmo/modules/vlmo_utils.py) | 哪些任务被启用？指标、优化器和学习率怎样安排？ |
| 10 | [my_metrics.py](../UniLM/vlmo/vlmo/gadgets/my_metrics.py) | 准确率、平均 loss、VQA 分数怎样累计？ |

### 先记住这些记号

`B` 是一批样本数，`T` 是一句话的 token 数，`P` 是图片 patch 数，`D` 是特征维度，`H` 是注意力头数。`[B,T,D]` 可以理解为“B 句话 × 每句 T 个位置 × 每个位置 D 个数字”。

默认 Base 配置使用 `image_size=224`、`patch_size=16`、`max_text_len=40`、`D=768`。因此图片有 `14×14=196` 个 patch，加图像 CLS 是 **197 个图像 token**；文字是 **40 个 token**；图文拼接是 `[B,237,768]`。配置改成 384 像素时，patch 数也会变化，不能继续套用 237。

### 逐句拆 `Block.forward()`

1. `self.norm1(x)`：先稳定每个 token 的 768 维数值，输出形状不变。
2. `self.attn(...)`：同一层注意力同时接收 image、text 或 vl 序列。它先由 `qkv` 线性层产生 Q、K、V，再算 `[B,H,N,N]` 的 token 两两分数。
3. `mask`：有 PAD 的文字位置被填成 `-inf`，经 softmax 后不会成为被读取的内容。
4. `relative_position_bias`：在 softmax 前加到两两分数上，告诉模型 token 的相对位置。
5. `self.gamma_1 * ...`：LayerScale 调节新注意力信息的初始强度；`self.drop_path(...)` 训练时可能跳过残差分支。
6. `x = x + ...`：把注意力结果加回原输入，叫残差连接。此时形状仍为 `[B,N,D]`。
7. `modality_type == "image"`：所有 token 走 `mlp_imag`；`"text"`：走 `mlp_text`。
8. 图文 `vl` 且本层有 `mlp_vl`：整条图文序列走第三套 FFN；没有时在 `max_text_len` 处分成文字和图像两段，分别走自己的 FFN 再拼回去。

**注意**：多路专家指不同的 FFN 参数。这里没有按概率随机挑专家，也不是三套独立的完整 Transformer。

### 为什么 `infer_text()` 和 `infer_image()` 会算两套向量？

预训练的 ITC 既比较单模态 FFN 路径，也比较后段 VL-FFN 路径。`infer_text()` 先让所有层走文字专家，并保存每层结果；再从 VL-FFN 启用前的位置分叉，重走最后几层。`infer_image()` 对图片做相同处理，所以各返回 `cls_feats` 和 `cls_vlffn_feats`。`compute_itc()` 为两套向量分别算图找文、文找图交叉熵，再平均。

检索微调的 `infer_text_ft()`、`infer_image_ft()` 则只保留第一套向量。这里的 `_ft` 是 *fine-tuning*，不是又创建一个新的大模型。

### 从 1 个 batch 追到 loss

```text
Arrow 图片字节、caption
  → BaseDataset.get_image/get_text
  → collate: image [B,3,224,224], text_ids [B,40]
  → VLMo.infer_image / infer_text / infer
  → objectives.compute_itc / compute_itm_hardneg / compute_mlm
  → VLMo.training_step 把名称含 loss 的结果相加
  → Lightning 自动反向传播和更新参数
```

`compute_itc()` 的图文相似度矩阵通常是 `[B,B]`，正确图文位于对角线。`compute_itm_hardneg()` 先利用这张矩阵挑相似但不正确的图/文，再组成 **B 对正确、B 对错图、B 对错文**，因此 `itm_score` 最后看到约 `3B` 对图文。`compute_mlm()` 的 `-100` 标签表示该文字位置不计入补词损失。

### loss 怎样真的改动模型参数？

`training_step()` 只负责返回合并后的 loss。Lightning 随后自动调用反向传播，求出各参数的梯度，再用 `configure_optimizers()` 返回的优化器更新参数。这个函数最终进入 [vlmo_utils.py](../UniLM/vlmo/vlmo/modules/vlmo_utils.py) 的 `set_schedule()`。

`set_schedule()` 把参数按两个条件分成四组：普通权重/偏置或归一化权重，以及模型主干/下游分类头。偏置与归一化权重不做 weight decay；分类头的学习率是 `learning_rate × lr_mult`。再选 AdamW、Adam 或 SGD；学习率先 warmup，随后按 cosine 或 polynomial 曲线降低。`interval="step"` 表示每更新一次参数就推进一次学习率调度。

每个 epoch 结束时，`epoch_wrapup()` 从 [my_metrics.py](../UniLM/vlmo/vlmo/gadgets/my_metrics.py) 的指标容器取出累计值、记录日志、清零，最后生成 `val/the_metric` 供模型检查点选择。`compute()` 是“读取本轮结果”，`reset()` 是“准备下一轮”；这些统计值本身不参与梯度更新。

### 几个容易卡住的 Python/PyTorch 写法

| 代码 | 中文意思 |
|---|---|
| `x[:, 0]` | 所有样本的第 0 个 token，通常是 CLS |
| `torch.cat(..., dim=1)` | 沿 token 序列方向拼接文字和图像 |
| `.transpose(-2, -1)` 或 `.t()` | 把矩阵最后两个维度互换，用于 QK 点积或反向检索 |
| `.view(-1, V)` | 把批量和词位置合并，让每个词都参加同一个分类损失 |
| `getattr(module, name)` | 按字符串找到名为 `name` 的指标对象，例如训练/验证准确率 |
| `dist.all_gather(...)` | 多张 GPU 交换向量，增加对比学习的错误候选数量 |
| `@ex.named_config` | Sacred 中可在命令行按名字选用的一组配置覆盖项 |

## 八、阅读范围与运行边界

这些中文注释覆盖了**训练入口、默认与任务配置、数据读取和拼批、骨干与模态专家、主要损失、检索评估、优化器调度与指标汇总**。仓库其他数据转换脚本和图像增强的第三方实现尚未逐行扩写；第一次读 VLMo 不需要先进入这些文件。原版训练依赖旧版 PyTorch Lightning、timm、Transformers、Arrow 数据及多 GPU 配置，当前工作区做的是源码精读，**尚未运行原版 VLMo 训练**。
