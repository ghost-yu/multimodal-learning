# 四项目核心源码教学副本

这里的六个 `.py` 是对应本地仓库文件的**加注释副本**，用于在编辑器中边看实际代码边读中文解释。保留所有原始可执行语句及原有注释，在难点语句旁直接补充面向只有 Python 基础的读者的说明；文件顶部标出原文件路径。

建议先看每篇精读文档的原理和变量表，再打开对应 `.py` 按函数顺序阅读。遇到 `x`、`feat`、`embeds` 等反复出现的名字，优先看其在**当前函数**的上一次赋值及旁边写的形状；同名变量在不同函数中不是同一个对象。

| 项目 | 教学副本 | 原始文件 | 详细原理与变量表 |
|---|---|---|---|
| CLIP | [加载与分词入口](CLIP-clip-详注.py)、[BPE 分词器](CLIP-simple_tokenizer-详注.py)、[模型代码](CLIP-model-详注.py) | [clip.py](../../CLIP/clip/clip.py)、[simple_tokenizer.py](../../CLIP/clip/simple_tokenizer.py)、[model.py](../../CLIP/clip/model.py) | [CLIP 论文与代码解释](../../Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision/解释.md) |
| ViLT | [ViLT-vilt_module-详注.py](ViLT-vilt_module-详注.py) | [vilt_module.py](../../ViLT/vilt/modules/vilt_module.py) | [ViLT 精读](../ViLT-零基础源码逐行精读.md) |
| ALBEF | [ALBEF-model_pretrain-详注.py](ALBEF-model_pretrain-详注.py) | [model_pretrain.py](../../ALBEF/models/model_pretrain.py) | [ALBEF 精读](../ALBEF-零基础源码逐行精读.md) |
| VLMO | [VLMO-multiway_transformer-详注.py](VLMO-multiway_transformer-详注.py) | [multiway_transformer.py](../../UniLM/vlmo/vlmo/modules/multiway_transformer.py) | [VLMO 精读](../VLMO-零基础源码逐行精读.md) |

具体覆盖：CLIP 的下载/预处理/tokenize、BPE 合并、JIT 修补、ResNet/ViT/文字编码与权重重建；ViLT 的模型构造、`infer` 融合、任务路由和 Lightning 回调；ALBEF 的教师、队列、ITA/ITM/MLM、遮词和分布式汇总；VLMO 的 patch 嵌入、QKV 注意力、专家 FFN 路由及三种模型规格。上述六个原始文件中的辅助分支也给出就地说明。

**范围**：六个教学副本的每段核心流程、难懂语句和分支已加注释；未逐行覆盖四个仓库的其他文件，例如 ViLT 的 `objectives.py`、VLMO 的 `vlmo_module.py`/`objectives.py`、ALBEF 的 `xbert.py`。外部 PyTorch/timm/Lightning 的内部源码也不在此范围；在当前文件调用它们的输入、输出和作用已解释。

这些教学文件使用原项目各自的导入方式。CLIP 分词器按 `__file__` 寻找同目录的 `.gz` 词表，因此本教学副本用于阅读；实际调用请用原仓库里的 `CLIP/clip/simple_tokenizer.py`。如果单独运行，Python 可能找不到 `models`、`vilt`、`vlmo` 等包；要实际运行训练，请从原项目入口与原项目环境启动。这里验证了语法、AST 中可执行结构与原文件一致及本目录链接有效；没有运行完整模型训练。
