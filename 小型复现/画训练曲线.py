"""把 ViLT 和 ALBEF 教学模型的逐轮指标画成一张图。"""

# json 读取训练脚本保存的每轮数字。
import json
# Path 让脚本从任何工作目录运行时都能找到数据。
from pathlib import Path

# 使用不需要弹出图形窗口的绘图后端，适合在命令行保存 PNG。
import matplotlib
matplotlib.use("Agg")
# pyplot 提供画线、标题与保存图片的函数。
import matplotlib.pyplot as plt

# __file__ 是这个脚本的路径；parent 是“小型复现”目录。
ROOT = Path(__file__).resolve().parent
# 训练脚本把每轮结果存放在 checkpoints 中。
CHECKPOINTS = ROOT / "checkpoints"


def read_history(model_name: str) -> list[dict]:
    """读取一个模型从第 1 轮到最后一轮的指标。"""
    # 例如 vilt 对应 vilt_history.json。
    path = CHECKPOINTS / f"{model_name}_history.json"
    # 文件不存在说明还没有运行 真实数据小训练.py。
    if not path.is_file():
        # 提示先训练的具体命令，方便初学者排查。
        raise FileNotFoundError(f"缺少 {path}；请先运行 真实数据小训练.py --model {model_name}")
    # 读取 UTF-8 文本并解析 JSON 数组。
    history = json.loads(path.read_text(encoding="utf-8"))
    # 空数组无法画出任何学习曲线。
    if not history:
        # 提前给出清晰错误。
        raise ValueError(f"训练历史为空：{path}")
    # 每个元素包含 epoch、训练/验证损失与训练/验证准确率。
    return history


def main() -> None:
    """每个模型画准确率和损失，共四张小图。"""
    # 两列分别画 ViLT 与 ALBEF，两行分别画准确率与损失。
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    # 遍历两个教学模型，column 决定放在左列还是右列。
    for column, model_name in enumerate(("vilt", "albef")):
        # 读取该模型全部训练轮次的记录。
        history = read_history(model_name)
        # 把每轮编号提取成横坐标。
        epochs = [row["epoch"] for row in history]
        # 上排画匹配分类准确率；训练集与验证集用两种颜色。
        accuracy_axis = axes[0, column]
        # 训练集的准确率线显示模型是否越来越会“记住”训练图片。
        accuracy_axis.plot(epochs, [row["train_accuracy"] for row in history], label="train", color="#2563eb")
        # 验证集的准确率线显示陌生图片上的表现是否同步改善。
        accuracy_axis.plot(epochs, [row["validation_accuracy"] for row in history], label="validation", color="#ea580c")
        # 二分类正负数量相同，0.5 是随机猜测的大致参考线。
        accuracy_axis.axhline(0.5, linestyle=":", color="#64748b", linewidth=1, label="chance (0.5)")
        # 固定纵轴为 0 到 1，方便两个模型直接对照。
        accuracy_axis.set_ylim(0, 1.05)
        # 标明模型名和该图的指标。
        accuracy_axis.set_title(f"{model_name.upper()} - matching accuracy")
        # 仅左列显示纵轴名称，避免重复文字拥挤。
        if column == 0:
            accuracy_axis.set_ylabel("accuracy")
        # 图例指出哪条线属于训练集和验证集。
        accuracy_axis.legend(fontsize=8)
        # 下排画损失；它反映模型对自己的判断有多自信。
        loss_axis = axes[1, column]
        # 画训练损失。
        loss_axis.plot(epochs, [row["train_loss"] for row in history], label="train", color="#2563eb")
        # 画验证损失。
        loss_axis.plot(epochs, [row["validation_loss"] for row in history], label="validation", color="#ea580c")
        # ViLT 损失只含 ITM；ALBEF 损失还加上 ITC，不能跨列直接比数值。
        loss_axis.set_title(f"{model_name.upper()} - training objective")
        # 告诉读者横轴是训练轮数。
        loss_axis.set_xlabel("epoch")
        # 左列显示纵轴名称。
        if column == 0:
            loss_axis.set_ylabel("loss")
        # 在损失图也保留颜色说明。
        loss_axis.legend(fontsize=8)
    # 自动留出标题与图例的空间。
    figure.tight_layout()
    # 保存到与训练历史相同的目录。
    output = CHECKPOINTS / "training_curves.png"
    # 150 DPI 足够在笔记中清晰阅读。
    figure.savefig(output, dpi=150)
    # 关闭绘图对象，释放内存。
    plt.close(figure)
    # 打印图片位置，方便用户打开。
    print(f"曲线图已保存：{output}")


if __name__ == "__main__":
    # 只有直接运行此文件时才画图；被 import 时不会自动执行。
    main()
