"""用已下载的 Flickr8k 小样本训练 ViLT 或 ALBEF 教学模型。"""

# argparse 读取 --model、--epochs 等命令行选项。
import argparse
# json 用于把训练结果保存成容易阅读的文本。
import json
# Path 管理数据目录与模型文件路径。
from pathlib import Path
# random 固定随机选描述的顺序。
import random

# PyTorch 负责模型、优化器和自动求导。
import torch
# F 提供分类交叉熵。
from torch.nn import functional as F
# DataLoader 将 Dataset 的多条样本合成一个 batch。
from torch.utils.data import DataLoader

# 导入 ALBEF 的教学模型类。
from albef_tiny import TinyALBEF
# 导入真实 Flickr8k 数据集与词表函数。
from flickr8k_data import Flickr8kMini, build_vocab, read_manifest
# 导入 ViLT 的教学模型类。
from vilt_tiny import TinyViLT


def matching_batch(images: torch.Tensor, texts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ViLT 使用：把 B 对正确图文扩成 B 个正例和 B 个负例。"""
    # B 是本次读取的图片数量。
    batch_size = images.shape[0]
    # 只有一张图时错开文字仍是原文，不能构造负例。
    if batch_size < 2:
        # 提前给出清晰错误，避免产生错误标签。
        raise ValueError("图文匹配训练每批至少需要 2 张图片")
    # 同一批图片使用两次，前一半配正确文字、后一半配错误文字。
    pair_images = torch.cat((images, images), dim=0)
    # roll 将文字沿批量轴移动一位，产生错配文字。
    pair_texts = torch.cat((texts, texts.roll(shifts=1, dims=0)), dim=0)
    # 前 B 个标签为匹配 1，后 B 个标签为不匹配 0。
    labels = torch.cat((torch.ones(batch_size, dtype=torch.long, device=images.device), torch.zeros(batch_size, dtype=torch.long, device=images.device)))
    # 返回分类模型需要的三个张量。
    return pair_images, pair_texts, labels


def run_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None, model_name: str) -> tuple[float, float]:
    """跑完训练或验证的一轮，返回平均损失与匹配准确率。"""
    # 有优化器表示训练阶段；验证阶段传入 None。
    training = optimizer is not None
    # train/eval 切换模型的运行模式。
    model.train(training)
    # 记录按样本数加权后的总损失。
    loss_sum = 0.0
    # 记录预测正确的图文对数量。
    correct = 0
    # 记录参与计算的图文对总数。
    total = 0
    # 只有训练阶段才需要保存梯度。
    with torch.set_grad_enabled(training):
        # DataLoader 每次给出一批图片和文字。
        for images, texts in loader:
            # 只剩一张图片时无法构造不同图片的负例，跳过这张。
            if images.shape[0] < 2:
                continue
            # 把图片放到模型所在的设备。
            images = images.to(device)
            # 把文字放到同一个设备。
            texts = texts.to(device)
            # ALBEF 一批输入仍是 B 对正确图文，负例在模型内部构造。
            if model_name == "albef":
                # 模型返回全局对齐 ITC、细粒度匹配 ITM 和分类分数。
                itc_loss, itm_loss, logits = model(images, texts)
                # 两个训练目标等权相加。
                loss = itc_loss + itm_loss
                # 分类标签与模型内部的正负例顺序一致。
                labels = matching_batch(images, texts)[2]
            else:
                # ViLT 需要在模型外部先造好正负图文对。
                pair_images, pair_texts, labels = matching_batch(images, texts)
                # 共享 Transformer 输出每一对图文的两个分数。
                logits = model(pair_images, pair_texts)
                # 用真实标签计算图文匹配交叉熵。
                loss = F.cross_entropy(logits, labels)
            # 验证阶段没有优化器，不能修改模型参数。
            if training:
                # 清空上一批样本留下的梯度。
                optimizer.zero_grad()
                # 计算当前损失对所有可训练参数的梯度。
                loss.backward()
                # 用梯度更新一次参数。
                optimizer.step()
            # 当前批的图文对数等于分类标签个数。
            pair_count = labels.numel()
            # 把当前平均损失乘以样本数，以后再求整体平均。
            loss_sum += loss.item() * pair_count
            # argmax 选分类分数更高的类别，并统计预测正确数量。
            correct += (logits.argmax(dim=1) == labels).sum().item()
            # 累加这批参与评估的图文对数。
            total += pair_count
    # 返回本轮平均损失和准确率。
    return loss_sum / total, correct / total


def main(model_name: str, epochs: int, batch_size: int, lr: float, device_name: str) -> None:
    """加载真实样本、建词表、训练、验证并保存模型。"""
    # 固定 Python 随机种子，便于复查随机描述的选择。
    random.seed(7)
    # 固定 PyTorch 随机种子，便于复查模型初始化。
    torch.manual_seed(7)
    # 程序文件旁边的 real_data 是下载脚本保存数据的位置。
    data_root = Path(__file__).resolve().parent / "real_data" / "flickr8k-mini"
    # 从训练清单读取图片路径和五条描述。
    train_rows = read_manifest(data_root, "train")
    # 验证清单与训练清单分开读取。
    validation_rows = read_manifest(data_root, "validation")
    # 只用训练文字建立词表，避免把验证答案泄漏给模型。
    vocab = build_vocab(train_rows)
    # 图片统一为 64×64，文字统一长度为 20。
    image_size, patch_size, text_len = 64, 8, 20
    # 入门实验固定使用每张图的第一条描述，方便观察模型能否学会小训练集。
    train_data = Flickr8kMini(data_root, train_rows, vocab, text_len, image_size, training=False)
    # 验证 Dataset 固定使用每张图第一条描述。
    validation_data = Flickr8kMini(data_root, validation_rows, vocab, text_len, image_size, training=False)
    # batch_size 至少 2，才能在一个 batch 内构造错误图文配对。
    if batch_size < 2 or batch_size > min(len(train_data), len(validation_data)):
        # 当前下载的验证集很小，使用不大于其大小的批量。
        raise ValueError("batch_size 必须至少为 2，且不超过训练集和验证集的图片数")
    # 丢掉最后不足一个 batch 的训练数据，避免只剩一张图片。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # 验证按固定顺序读取；只要最后一批至少两张，剩余样本也参与评估。
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=False)
    # auto 有 CUDA 时用显卡；用户也可以指定 cpu 避免显卡被别的程序占满。
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    # 把设备名称包装成 PyTorch 设备对象。
    device = torch.device(selected_device)
    # 两个模型共享尺寸配置，但保留各自不同的融合结构。
    model_args = {"image_size": image_size, "patch_size": patch_size, "text_len": text_len, "vocab_size": len(vocab)}
    # 根据命令行选择 ViLT 或 ALBEF 教学模型。
    model = TinyViLT(**model_args) if model_name == "vilt" else TinyALBEF(**model_args)
    # 把模型参数移到与输入相同的计算设备。
    model = model.to(device)
    # AdamW 根据梯度更新参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 告诉读者实际样本量、词表规模和运行设备。
    print(f"模型 {model_name} | 训练图片 {len(train_data)} | 验证图片 {len(validation_data)} | 词表 {len(vocab)} | 设备 {device}")
    # history 按轮保存曲线数据，方便区分“记住训练集”和“学会泛化”。
    history = []
    # 重复遍历整个小训练集指定次数。
    for epoch in range(1, epochs + 1):
        # 有优化器时这一轮会实际更新模型参数。
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer, model_name)
        # 没有优化器时只计算验证结果，不更新参数。
        val_loss, val_accuracy = run_epoch(model, validation_loader, device, None, model_name)
        # 同时打印训练与验证指标，观察是否只记住训练数据。
        print(f"第 {epoch:02d}/{epochs} 轮 | 训练损失 {train_loss:.3f} 准确率 {train_accuracy:.0%} | 验证损失 {val_loss:.3f} 准确率 {val_accuracy:.0%}")
        # 把本轮四项数字存进一个字典，之后可画图或查最佳轮次。
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": val_loss, "validation_accuracy": val_accuracy})
    # 将两个模型分别保存在 checkpoints 目录。
    output_dir = Path(__file__).resolve().parent / "checkpoints"
    # 不存在就创建目录。
    output_dir.mkdir(exist_ok=True)
    # 文件名包含模型名，避免互相覆盖。
    checkpoint_path = output_dir / f"{model_name}_flickr8k_mini.pt"
    # 保存参数、词表和结构设置，以后才能按同样方式加载。
    torch.save({"model": model.state_dict(), "vocab": vocab, "model_args": model_args}, checkpoint_path)
    # 另存纯文本结果，便于不懂 PyTorch 的读者查看。
    metrics_path = output_dir / f"{model_name}_metrics.json"
    # 把最后一轮指标写入 JSON 文件。
    metrics_path.write_text(json.dumps({"model": model_name, "train_images": len(train_data), "validation_images": len(validation_data), "epochs": epochs, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": val_loss, "validation_accuracy": val_accuracy}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 另一份 JSON 保留每一轮的数字，便于画完整学习曲线。
    history_path = output_dir / f"{model_name}_history.json"
    # indent 使初学者直接打开文件也容易阅读。
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    # 打印模型保存位置。
    print(f"模型已保存：{checkpoint_path}")


if __name__ == "__main__":
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="Flickr8k 小样本真实数据训练")
    # 必须明确选择要训练 ViLT 还是 ALBEF。
    parser.add_argument("--model", choices=("vilt", "albef"), required=True)
    # 轮数控制遍历整个小训练集多少遍。
    parser.add_argument("--epochs", type=int, default=80)
    # 每批四张图，两张以上即可构造正负配对。
    parser.add_argument("--batch-size", type=int, default=4)
    # 学习率控制每次更新的幅度。
    parser.add_argument("--lr", type=float, default=0.001)
    # 显卡内存不足时可加 --device cpu。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 解析命令行文字。
    args = parser.parse_args()
    # 检查轮数为正数，避免生成未训练的权重。
    if args.epochs < 1:
        # 把错误直接写成可理解的提示。
        parser.error("--epochs 必须大于 0")
    # 调用完整训练流程。
    main(args.model, args.epochs, args.batch_size, args.lr, args.device)
