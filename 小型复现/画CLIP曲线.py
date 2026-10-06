"""画出 CLIP 教学模型已见图片与验证图片的双向检索曲线。"""

# json 读取训练脚本保存的每轮数字。
import json
# Path 定位历史数据和输出图片。
from pathlib import Path

# Agg 后端只保存 PNG，不要求打开桌面绘图窗口。
import matplotlib
matplotlib.use("Agg")
# pyplot 负责曲线、坐标轴和图例。
import matplotlib.pyplot as plt

# 训练指标文件放在脚本旁边的 checkpoints 目录。
OUTPUT_ROOT = Path(__file__).resolve().parent / "checkpoints"
# JSON 数组里每个元素对应一轮训练。
history = json.loads((OUTPUT_ROOT / "clip_history.json").read_text(encoding="utf-8"))
# 横轴显示第几轮训练。
epochs = [row["epoch"] for row in history]
# 一行两列：左边图找文，右边文找图。
figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
# 两个方向使用相同绘图规则，仅字段名不同。
for axis, train_key, validation_key, title in (
    (axes[0], "train_probe_image_to_text_r1", "validation_image_to_text_r1", "image to text"),
    (axes[1], "train_probe_text_to_image_r1", "validation_text_to_image_r1", "text to image"),
):
    # 蓝线是训练清单中前 16 张已见图片的 R@1。
    axis.plot(epochs, [row[train_key] for row in history], label="seen 16 images", color="#2563eb")
    # 橙线是未参与更新的 16 张验证图片的 R@1。
    axis.plot(epochs, [row[validation_key] for row in history], label="validation 16 images", color="#ea580c")
    # 16 个候选里随机猜中一次的机会为 1/16。
    axis.axhline(1 / 16, color="#64748b", linestyle=":", label="chance (1/16)")
    # 两个方向均把 R@1 纵轴固定到 0～1，方便对照。
    axis.set_ylim(0, 1)
    # 标明当前方向。
    axis.set_title(title)
    # 横轴是训练轮次。
    axis.set_xlabel("epoch")
    # 每个小图放一份说明，避免读者把两组线弄反。
    axis.legend(fontsize=8)
# 左边图标明共同的纵轴含义。
axes[0].set_ylabel("Recall@1")
# 自动留出标题与图例需要的空间。
figure.tight_layout()
# 图片文件名说明画的是 CLIP 的检索指标。
path = OUTPUT_ROOT / "clip_retrieval_curves.png"
# 保存清晰的 150 DPI PNG。
figure.savefig(path, dpi=150)
# 释放绘图对象占用的内存。
plt.close(figure)
# 在终端显示生成的位置。
print(f"CLIP 曲线图已保存：{path}")
