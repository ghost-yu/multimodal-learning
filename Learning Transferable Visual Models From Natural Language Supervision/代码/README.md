# CLIP 代码

官方仓库位于工作区根目录的 [`CLIP`](../../CLIP)。本专属论文目录不复制整个仓库，避免产生两份容易失同步的源码；阅读入口和文件映射集中在 [`../解释.md`](../解释.md)。

核心文件：

- [`clip/clip.py`](../../CLIP/clip/clip.py)：模型加载、图片预处理和 `tokenize()`。
- [`clip/simple_tokenizer.py`](../../CLIP/clip/simple_tokenizer.py)：字节级 BPE 分词。
- [`clip/model.py`](../../CLIP/clip/model.py)：ViT、ModifiedResNet、文本 Transformer 和相似度计算。
- [`clip/__init__.py`](../../CLIP/clip/__init__.py)：包顶层导出。

当前这些原始文件只增加了中文注释；可执行代码结构保持不变。
