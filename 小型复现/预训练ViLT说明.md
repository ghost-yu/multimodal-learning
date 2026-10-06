# 使用公开 ViLT 权重做小规模检索头微调

入口：[预训练ViLT小微调.py](预训练ViLT小微调.py)。它与 [vilt_tiny.py](vilt_tiny.py) 是两个不同模型：前者加载原版 ViLT-B/32 Flickr30k 权重，后者是从零训练的教学缩小版，权重不能互换。

## 准备

本地模型文件位于 `小型复现/pretrained/vilt-flickr30k/`。当前工作区已经下载好权重和处理器。权重来源是 [dandelin/vilt-b32-finetuned-flickr30k](https://huggingface.co/dandelin/vilt-b32-finetuned-flickr30k)，模型文件约 446 MB，另需 `config.json`、`preprocessor_config.json`、`special_tokens_map.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt`。本项目使用已下载的 64 张训练图、16 张验证图。

在 `多模态` 目录运行：

```powershell
py -3.13 小型复现/预训练ViLT小微调.py
```

当前环境第一次导入 Transformers 可能需要几十秒。程序依次完成：加载原权重 → 微调前验证 → 冻结 ViLT 骨干、只更新 `rank_output` 检索头一轮 → 微调后验证 → 保存检索头。

## 一行行看训练流程

1. `read_rows()` 读真实图片与描述的 JSON 清单；`read_batch()` 逐张打开图片，固定每张图第一条英文描述。
2. `score_pairs()` 在一个 batch 内保留正确配对，再把文字错开一位生成负例。`ViltProcessor` 负责图片缩放、归一化、英文分词和补齐；`ViltForImageAndTextRetrieval` 输出每对图文的检索分数。
3. `evaluate()` 比较每张图片的“正确描述分数”和“一条错配描述分数”。这是**二选一排序检查**。
4. `evaluate_gallery()` 形成 `[16,16]` 分数表：每张图都要在 16 条文字里找到自己对应的描述；反过来，每句话也要在 16 张图里找到自己的图片。这分别是图找文、文找图 **Recall@1**。
5. `model.vilt.requires_grad_(False)` 冻结大型骨干。`optimizer` 只接收 `model.rank_output.parameters()`，因此一轮训练仅更新最后的线性检索头。
6. `F.softplus(negative-positive)` 在错误描述分数高于正确描述时变大，反向传播会推动正确分数更高。最终仅保存检索头到 `checkpoints/vilt_pretrained_rank_head.pt`。

## 本机实测

| 指标 | 微调前 | 微调一轮后 |
|---|---:|---:|
| 一条正例对一条负例的排序正确率 | 100% | 100% |
| 16 选 1 图找文 R@1 | 94% | 94% |
| 16 选 1 文找图 R@1 | 100% | 100% |

这份权重原本就针对 Flickr30k 做过检索微调。Flickr30k 与 Flickr8k 有继承关系，**目前没有核实这些验证图片是否出现在原权重的训练集中**；因此这里的高分只用于确认本地推理与微调流程，不能当成独立测试集的泛化成绩。16 张图的样本量也非常小，一轮头部微调没有观察到 R@1 提升。

与 [从零训练教学模型的检索结果](真实数据训练.md) 对照时，只能看出“已有大规模训练的权重在这份样本上明显更强”，不能据此公平比较模型架构。想做正规实验，需要核对原模型训练集、使用严格隔离的测试集和标准数据划分。

[通用预训练 ViLT 对照](通用预训练ViLT说明.md)另用只完成 MLM+ITM 通用预训练、没有做 Flickr30k 检索微调的权重，在同样 16 张图片上测得微调前图找文 100%、文找图 94%。两个结果都很高，进一步说明这个 16 候选的小验证集太容易、样本太少，不能拿微小差异判断哪个训练阶段更好。
