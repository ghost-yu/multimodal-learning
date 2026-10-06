# 08｜Uni-Perceiver 系列：让不同任务使用相似的输入和预测接口

> 本地论文：[原版](../papers/Uni-Perceiver.pdf)、[MoE](../papers/Uni-Perceiver-MoE.pdf)、[v2](../papers/Uni-Perceiver-v2.pdf)。本地代码：[Uni-Perceiver](../Uni-Perceiver)。该仓库说明了原版和 MoE 的训练/评估；阅读时不要默认它包含 v2 的完整代码。

## 一、先把它看成“多任务软件框架”

前面几篇常围绕图文两种模态、几个任务。Uni-Perceiver 希望让图片、视频、文字等输入经一套尽量统一的处理流程，用同一类模型解决分类、检索、caption 等任务。代码因而比 CLIP 大很多：除了模型，还有配置、数据封装、预测器、损失、训练器、评估器。

读这个项目的关键不是立刻看懂每个配置文件，而是追踪一个统一问题：**输入是什么，候选目标是什么，模型怎样给每个候选打分？**

## 二、总地图

| 位置 | 工作 |
|---|---|
| [main.py](../Uni-Perceiver/main.py) | 读取配置，启动训练或测试 |
| [configs](../Uni-Perceiver/configs) | 每个模型大小与任务组合的参数 |
| [uniperceiver/datasets](../Uni-Perceiver/uniperceiver/datasets) | 读取不同任务的数据，组织统一 batch 字典 |
| [uniperceiver/modeling/embedding](../Uni-Perceiver/uniperceiver/modeling/embedding) | 把图片、视频、文字转成 token 向量 |
| [unified_transformer.py](../Uni-Perceiver/uniperceiver/modeling/meta_arch/unified_transformer.py) | 整合输入、候选、编码器、预测器和损失 |
| [embed_cls_predictor.py](../Uni-Perceiver/uniperceiver/modeling/predictor/embed_cls_predictor.py) | 用表示向量形成分类或检索分数 |
| [uniperceiver/task_moe](../Uni-Perceiver/uniperceiver/task_moe) | MoE 专家和门控逻辑 |
| [uniperceiver/engine](../Uni-Perceiver/uniperceiver/engine) | 训练/测试循环、日志、调度 |

建议先看 [data_structure.md](../Uni-Perceiver/data/data_structure.md)，理解统一 batch 字典字段，再进入核心模型。

## 三、从 `main()` 到模型

[main.py:105](../Uni-Perceiver/main.py) 是总入口。它先通过配置知道任务类型、模型大小、数据位置，再建立训练器和模型。配置并不是“论文算法本身”，只是告诉同一套程序该选哪些组件。初学者可选一个分类或检索配置追踪，其他几十份 YAML 暂时跳过。

[MultiTaskTransformerEncoder._forward():262](../Uni-Perceiver/uniperceiver/modeling/meta_arch/unified_transformer.py) 收到 `batched_inputs` 字典。这个字典里不仅有输入样本，还有 `target_sample_list` 和 `shared_target_sets`，代表要比较的候选。比如图像分类时输入是图片，候选可理解为类别描述；图文检索时输入是图片，候选是一组文字。代码分别调用 `_forward_data()` 把输入和候选编码，再通过 `loss_prepare()` 得到分数，训练时按当前任务选损失。

## 四、`_forward_data()` 四步

[unified_transformer.py:315](../Uni-Perceiver/uniperceiver/modeling/meta_arch/unified_transformer.py) 是最值得逐句读的函数：

1. `_tokenize()` 检查 `data['modality']`。若是 `image` 或 `video`，调用视觉 embedding；若是 `text`，调用文本 embedding。这里的“tokenize”是广义的，即把各种原始输入转成 token 表示。
2. `_forward_unfused_encoders()` 可先使用某个模态自己的编码器。`unfused` 意味着这时尚未进入共享融合编码器。
3. `preprocess()` 将各输入的 token、mask、任务信息整理为共享编码器所需格式，可能包含 prompt 或标签提示。
4. `fused_encoder()` 执行共享编码，`postprocess()` 整理输出。最终 `_forward_data()` 返回特征，供输入和候选比较。

每一步都在**变换数据表示**，并非每一步都算损失。损失要等到输入与候选都得到表示，产生可比较的分数后才计算。

## 五、预测器在做什么

[EmbedClsAsRetrievalPredictor.forward():191](../Uni-Perceiver/uniperceiver/modeling/predictor/embed_cls_predictor.py) 收到编码结果后，组织输入与候选的表示并输出 logits。这里 `cls` 是全局汇总 token；`retrieval` 表示可把任务视为“从候选中检索一个答案”。同一个概念可覆盖分类和图文匹配，但不同任务仍可能有自己的标签、损失和后处理。

`_forward()` 里训练时按 `task_info['task_name']` 找对应损失，评估时返回分数供 evaluator 计算指标。对于初学者，先弄懂**分数矩阵的每一行、每一列分别对应什么**，再看温度参数、队列等细节。

## 六、MoE 版本增加了什么

MoE 是 mixture of experts，指模型有多个可选择的专家网络。这里 [task_moe/gate.py](../Uni-Perceiver/uniperceiver/task_moe/gate.py) 决定路由权重，[task_moe/experts.py](../Uni-Perceiver/uniperceiver/task_moe/experts.py) 实现专家，[transformer_encoder_moe_layer.py](../Uni-Perceiver/uniperceiver/modeling/layers/transformer_encoder_moe_layer.py) 把它们接进 Transformer 层。注意它和 VLMo 的“按图像/文字类型选择 FFN”不是完全相同的实现；先读原版输入与候选比较，再读路由策略。

## 七、为什么不建议从头复现整系列

这个仓库支持多模态、多任务、许多数据集和规模；每个配置还可能依赖外部数据、预训练权重、分布式训练。对大三初学者，先挑**一个**任务，例如图像分类或图文检索，看数据结构、一次前向和指标，已经能学到最有用的软件设计思想。系列里 v2 的论文可先关注它新增的任务范围与结构，不能在本地原版代码里强找不存在的实现。

**本项目完成标准**：能解释 `batched_inputs` 的输入与候选为何都要编码；能指出 `_tokenize()`、`fused_encoder()`、predictor、loss 各在哪一层；能区分原版代码与 MoE 扩展。
