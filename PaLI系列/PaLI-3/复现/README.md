# PaLI-3 检查记录

当前没有完整 PaLI-3 训练复现。这里的脚本仅核对 SigLIP 的数学损失与偏置梯度，使用纯 Python 标准库，不调用原 JAX 训练器，也不下载模型权重。

在项目根目录执行：

```powershell
py -3.13 -X utf8 "PaLI系列\PaLI-3\复现\scripts\检查SigLIP损失.py"
```

`py -3.13` 选择 Python 3.13；`-X utf8` 启用 UTF-8；引号中的路径为脚本。实际结果保存到 [SigLIP损失检查.json](outputs/SigLIP损失检查.json)。检查覆盖 B=2 的 1/B 归约、稳定负对数 sigmoid、两个偏置位置的解析梯度与中心差分；不覆盖原训练器的数据、分布式执行、优化器与模型性能。
