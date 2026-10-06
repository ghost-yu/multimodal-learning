# CoCa 复现记录

## 已执行范围

使用真实本地 `CoCa`、`CoCaTask` 与 `CoCaLoss`，构建 1 层视觉、1 层文字、1 层多模态解码的小网络。CPU、宽度 32、人工词表 32、图片 32×32、两对固定随机图文。进行了普通/融合 CE 对照、20 次优化和生成接口检查。

模拟图文不携带真实语义标签，不能评价自然语言 caption 能力，也不能报告论文任务性能。

## 环境

Windows；Python 3.13.5；PyTorch 2.6.0+cu124，执行选择 CPU。独立目录 `.venv` 使用 `--system-site-packages` 复用现有 PyTorch，局部安装 torchvision 0.21.0、timm 1.0.30、ftfy 6.3.1、wcwidth 0.9.2。transformers 从系统环境使用 5.16.1。没有修改系统 PyTorch。

最初缺 torchvision，安装后缺 ftfy 所需 wcwidth；补齐后重新运行。默认配置的清华源没有匹配包，改用 PyPI 官方源取得 Windows Python 3.13 wheel。

## 运行

在本项目根目录 PowerShell 中：

```powershell
& '复现/.venv/Scripts/python.exe' -X utf8 '复现/scripts/检查小型CoCa.py'
```

`&` 是 PowerShell 调用运算符，用指定路径的 Python 执行脚本；`-X utf8` 明确启用 UTF-8。脚本使用 `sys.path` 优先读取本仓库 `src`，避免误导入其他 OpenCLIP 安装。

[脚本](scripts/检查小型CoCa.py) 已注释每个操作、配置名和重要形状。正式结果在 [小型CoCa检查.json](outputs/小型CoCa检查.json)，以实际记录为准。

## 结果解释

`image_features/text_features=[2,32]`；`logits=[2,8,32]`；右移目标含 EOS，补齐为 -100。第一次执行 20 步，总损失由 10.638439 降到 0.282685，普通 CE 与融合 CE 在容差内一致，得到有限且非零梯度。

损失下降说明能拟合同一小批模拟数据，不能说明泛化。生成返回人工编号只验证接口，不能当作有意义的句子。完整原论文预训练与真实数据评估仍未复现。
