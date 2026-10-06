"""用 64 张 Flickr8k 训练图片从零训练 CLIP 教学版，并测双向检索。"""

# argparse 负责读取用户输入的训练轮数、批量大小和学习率。
import argparse
# json 把指标保存为可以直接打开的文本。
import json
# Path 管理图片、代码和模型文件的路径。
from pathlib import Path
# random 控制每张图随机选哪一条训练描述。
import random

# PyTorch 负责自动求导、张量计算和保存权重。
import torch
# F.cross_entropy 计算 CLIP 的双向对比损失。
from torch.nn import functional as F
# DataLoader 把多张图片合并为一个训练 batch。
from torch.utils.data import DataLoader

# 只包含双编码器结构的 CLIP 教学模型。
from clip_tiny import TinyCLIP
# 复用前面项目的 Flickr8k 图片读取、分词和词表代码。
from flickr8k_data import Flickr8kMini, build_vocab, read_manifest

# 所有文件路径都以脚本所在的“小型复现”目录为起点。
ROOT = Path(__file__).resolve().parent
# 下载脚本保存的 64 张训练图片和 16 张验证图片位于此处。
DATA_ROOT = ROOT / "real_data" / "flickr8k-mini"
# 训练后的权重与曲线数据放在同一个目录。
OUTPUT_ROOT = ROOT / "checkpoints"


def clip_loss(scores: torch.Tensor) -> torch.Tensor:
    """对 [B,B] 相似度矩阵的行和列分别计算交叉熵。"""
    # B 是一批图文对的数量；矩阵必须有 B 行和 B 列。
    batch_size = scores.shape[0]
    # 第 i 张图对应第 i 句话，所以目标编号是 0、1、...、B-1。
    targets = torch.arange(batch_size, device=scores.device)
    # 把每行看作“这张图从 B 句话里选正确描述”。
    image_to_text = F.cross_entropy(scores, targets)
    # 转置后把每行看作“这句话从 B 张图里选正确图片”。
    text_to_image = F.cross_entropy(scores.T, targets)
    # 两个方向权重相同，这就是 CLIP 的对称对比学习目标。
    return (image_to_text + text_to_image) / 2


def train_one_epoch(model: TinyCLIP, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """训练集遍历一次，并返回所有 batch 的平均损失。"""
    # 训练模式允许模型执行需要的训练行为。
    model.train()
    # 把每一批的损失按图片数累加。
    total_loss = 0.0
    # 记录实际训练的图文对数量。
    total_pairs = 0
    # DataLoader 逐批交出 [B,3,64,64] 图片和 [B,20] 文字 ID。
    for images, texts in loader:
        # 模型与输入必须放在同一个设备，GPU 可用时通常是 cuda。
        images = images.to(device)
        # 文字整数张量也移到同一设备。
        texts = texts.to(device)
        # 双编码器产生 [B,B] 相似度矩阵；每个元素都是一对图文的分数。
        scores = model(images, texts)
        # 正确图文在对角线，其他位置充当本批负例。
        loss = clip_loss(scores)
        # 清空上一批留在参数上的梯度。
        optimizer.zero_grad()
        # 自动求出损失对两个编码器和温度参数的梯度。
        loss.backward()
        # 根据梯度更新所有可训练参数。
        optimizer.step()
        # 一个 batch 的平均损失乘以样本数，最后才可算整轮平均。
        total_loss += loss.item() * images.shape[0]
        # 本批有多少对正确图文，就累加多少。
        total_pairs += images.shape[0]
    # 返回这一轮的平均双向对比损失。
    return total_loss / total_pairs


def retrieval_r1(model: TinyCLIP, dataset: Flickr8kMini, batch_size: int, device: torch.device) -> tuple[float, float]:
    """在固定的 16 张图与 16 条描述中计算双向 Recall@1。"""
    # 按清单顺序取数据；第 i 张图仍对应第 i 句话。
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 图像向量按批暂存于列表。
    image_chunks = []
    # 文本向量也按同样顺序暂存。
    text_chunks = []
    # 评估模式关闭训练期的随机行为。
    model.eval()
    # 检索只做前向计算，不需要保存梯度。
    with torch.no_grad():
        # 逐批读取独立的图像和文字。
        for images, texts in loader:
            # encode_image 仅处理图片；这正是双编码器可提前编码的优点。
            image_chunks.append(model.encode_image(images.to(device)))
            # encode_text 仅处理文字，不需要重复算图片。
            text_chunks.append(model.encode_text(texts.to(device)))
        # 将多个 [B,E] 批拼回完整的 [N,E] 图像向量表。
        image_vectors = torch.cat(image_chunks, dim=0)
        # 同样得到 [N,E] 的文字向量表。
        text_vectors = torch.cat(text_chunks, dim=0)
        # 一次矩阵乘法就能给所有 N×N 对图文打分。
        scores = image_vectors @ text_vectors.T
        # 对角线的编号 0、1、...、N-1 是每个查询的正确答案。
        targets = torch.arange(len(dataset), device=device)
        # 每行最高分对应一张图找出的描述；统计正确比例。
        image_to_text = (scores.argmax(dim=1) == targets).float().mean().item()
        # 每列最高分对应一句话找出的图片；统计正确比例。
        text_to_image = (scores.argmax(dim=0) == targets).float().mean().item()
    # 返回两个方向的 Recall@1；16 个候选的随机机会约为 6.25%。
    return image_to_text, text_to_image


def main(epochs: int, batch_size: int, lr: float, device_name: str) -> None:
    """准备数据、训练模型、评估检索并保存实验结果。"""
    # 固定 Python 随机数，尽量复现每张图随机抽取的 caption。
    random.seed(7)
    # 固定模型初始化和 DataLoader 打乱顺序的 PyTorch 随机数。
    torch.manual_seed(7)
    # 训练清单只供参数更新和建词表使用。
    train_rows = read_manifest(DATA_ROOT, "train")
    # 验证清单不参与词表建立或参数更新。
    validation_rows = read_manifest(DATA_ROOT, "validation")
    # 只统计训练集的英文词，避免把验证描述里的新词提前泄漏给模型。
    vocab = build_vocab(train_rows)
    # 保存模型尺寸配置，以便以后按相同结构重新加载权重。
    model_args = {"vocab_size": len(vocab), "image_size": 64, "patch_size": 8, "text_len": 20, "width": 64, "embed_dim": 64}
    # 训练时，每次读取图片随机选其五条描述中的一条。
    train_data = Flickr8kMini(DATA_ROOT, train_rows, vocab, text_len=20, image_size=64, training=True)
    # 只取训练集前 16 张作为“已见图片”的固定检索检查组。
    train_probe = Flickr8kMini(DATA_ROOT, train_rows[:16], vocab, text_len=20, image_size=64, training=False)
    # 验证时固定第一条描述，保证每轮对比的是同一批答案。
    validation_data = Flickr8kMini(DATA_ROOT, validation_rows, vocab, text_len=20, image_size=64, training=False)
    # 至少 2 个样本才能构造 batch 内负例，batch 也不能超过训练图数。
    if batch_size < 2 or batch_size > len(train_data):
        # 在开始训练前就说明参数输入错误。
        raise ValueError("batch_size 必须在 2 到训练图片数之间")
    # 打乱训练图片；不足一个 batch 的尾批丢掉，避免只剩单个样本。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # auto 优先使用 CUDA；也允许明确选择 CPU。
    chosen_device = ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    # 用 PyTorch 的设备对象统一管理张量位置。
    device = torch.device(chosen_device)
    # 创建两个编码器，参数从随机值开始，没有加载官方 CLIP 权重。
    model = TinyCLIP(**model_args).to(device)
    # AdamW 同时更新图像编码器、文字编码器和温度参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 训练之前先测一次，知道随机初始化模型的起点。
    before_train = retrieval_r1(model, train_probe, batch_size, device)
    # 同样测验证图片，后面用相同 16 张图片复测。
    before_validation = retrieval_r1(model, validation_data, batch_size, device)
    # 打印数据规模与计算设备。
    print(f"CLIP 教学版 | 训练 {len(train_rows)} 图 | 验证 {len(validation_rows)} 图 | 词表 {len(vocab)} | {device}")
    # 每个方向分别报告训练前 16 选 1 的检索正确率。
    print(f"训练前 | 已见图 16 选 1 图找文 {before_train[0]:.0%} 文找图 {before_train[1]:.0%} | 验证图 图找文 {before_validation[0]:.0%} 文找图 {before_validation[1]:.0%}")
    # history 保存每轮指标，方便检查模型是否只记住训练图片。
    history = []
    # 从第 1 轮训练到用户指定的最后一轮。
    for epoch in range(1, epochs + 1):
        # 一轮会遍历训练集，每批随机选择一种描述。
        loss = train_one_epoch(model, train_loader, optimizer, device)
        # 用 16 张已见图片组成的固定候选集计算双向 R@1。
        train_i2t, train_t2i = retrieval_r1(model, train_probe, batch_size, device)
        # 再在 16 张没参加训练的验证图上计算相同指标。
        val_i2t, val_t2i = retrieval_r1(model, validation_data, batch_size, device)
        # 保存这轮的损失和四个检索指标。
        history.append({"epoch": epoch, "train_batch_loss": loss, "train_probe_image_to_text_r1": train_i2t, "train_probe_text_to_image_r1": train_t2i, "validation_image_to_text_r1": val_i2t, "validation_text_to_image_r1": val_t2i})
        # 每轮显示结果，便于观察训练集与验证集是否分叉。
        print(f"第 {epoch:02d}/{epochs} 轮 | 对比损失 {loss:.3f} | 已见图 R@1 {train_i2t:.0%}/{train_t2i:.0%} | 验证图 R@1 {val_i2t:.0%}/{val_t2i:.0%}")
    # 没有输出目录时自动创建，有目录时继续沿用。
    OUTPUT_ROOT.mkdir(exist_ok=True)
    # 保存参数、词表和结构配置；三者必须一起使用才能重新推理。
    torch.save({"model": model.state_dict(), "vocab": vocab, "model_args": model_args}, OUTPUT_ROOT / "clip_flickr8k_mini.pt")
    # 将训练过程保存为文本数组，不懂 PyTorch 也可查看每轮数字。
    (OUTPUT_ROOT / "clip_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    # 另存一份简短总结，明确模型是从零训练的教学版。
    summary = {"model": "TinyCLIP from scratch", "train_images": len(train_rows), "validation_images": len(validation_rows), "epochs": epochs, "batch_size": batch_size, "before_validation_image_to_text_r1": before_validation[0], "before_validation_text_to_image_r1": before_validation[1], "after_validation_image_to_text_r1": val_i2t, "after_validation_text_to_image_r1": val_t2i}
    # indent=2 让 JSON 排版适合人工阅读。
    (OUTPUT_ROOT / "clip_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 打印权重文件名，方便下一步加载和对比。
    print(f"模型和逐轮指标已保存至：{OUTPUT_ROOT}")


if __name__ == "__main__":
    # 创建命令行选项读取器。
    parser = argparse.ArgumentParser(description="在 Flickr8k 小样本上训练 CLIP 教学版")
    # 默认训练 80 轮，与前面的 ViLT/ALBEF 教学实验规模相近。
    parser.add_argument("--epochs", type=int, default=80)
    # 每批 8 对图文，形成 8×8 相似度矩阵。
    parser.add_argument("--batch-size", type=int, default=8)
    # 学习率决定 AdamW 每次更新的幅度。
    parser.add_argument("--lr", type=float, default=0.001)
    # 显卡不方便使用时可写 --device cpu。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 将命令行文字转换成字段值。
    args = parser.parse_args()
    # 至少训练一轮，才能得到真正训练过的权重。
    if args.epochs < 1:
        # 说明允许的范围，而不是让模型空跑。
        parser.error("--epochs 必须大于 0")
    # 开始下载好的真实图片小训练。
    main(args.epochs, args.batch_size, args.lr, args.device)
