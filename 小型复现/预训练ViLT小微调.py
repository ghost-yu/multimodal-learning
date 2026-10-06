"""用 ViLT 官方 Flickr30k 权重做 Flickr8k 小样本检索头微调。"""

# argparse 负责读取轮数与每批图片数量。
import argparse
# json 用来读取数据清单并保存实验结果。
import json
# Path 负责定位本地权重与图片文件。
from pathlib import Path

# Pillow 负责打开真实 JPEG 图片。
from PIL import Image
# PyTorch 负责损失、反向传播与优化器。
import torch
# softplus 构造“正确描述分数高于错配描述”的排序损失。
from torch.nn import functional as F
# Transformers 提供 ViLT 的原版结构和配套图片/文字预处理器。
from transformers import ViltForImageAndTextRetrieval, ViltProcessor

# 所有文件路径都以本脚本的位置为基准。
ROOT = Path(__file__).resolve().parent
# Flickr8k 小样本由下载脚本保存于此。
DATA_ROOT = ROOT / "real_data" / "flickr8k-mini"
# 官方 Flickr30k 微调权重和处理器文件保存于此。
WEIGHT_ROOT = ROOT / "pretrained" / "vilt-flickr30k"


def read_rows(split: str) -> list[dict]:
    """读取训练或验证清单，并检查它确实存在。"""
    # 根据 split 选 train.json 或 validation.json。
    path = DATA_ROOT / f"{split}.json"
    # 没有清单时提醒先执行下载脚本。
    if not path.is_file():
        # 清晰报出缺失文件完整路径。
        raise FileNotFoundError(f"缺少 {path}；请先运行 下载Flickr8k样本.py")
    # 以 UTF-8 打开 JSON 文件。
    with path.open("r", encoding="utf-8") as handle:
        # 把 JSON 数组变成 Python 列表。
        return json.load(handle)


def read_batch(rows: list[dict]) -> tuple[list[Image.Image], list[str]]:
    """读取一批图片，并取每张图的第一条描述。"""
    # 图片对象组成列表，稍后统一交给 ViltProcessor。
    images = []
    # 每张图片对应一条正确描述。
    captions = []
    # 逐条读取清单。
    for row in rows:
        # 把相对路径拼成真实图片文件位置。
        image_path = DATA_ROOT / row["image"]
        # 用 with 确保打开后及时关闭磁盘文件。
        with Image.open(image_path) as source:
            # copy 保留像素，让文件关闭后图片仍可供处理器使用。
            images.append(source.convert("RGB").copy())
        # 固定第一条文字，方便前后两次验证使用完全相同的数据。
        captions.append(row["captions"][0])
    # 返回一一对应的图片与文字列表。
    return images, captions


def score_pairs(model: ViltForImageAndTextRetrieval, processor: ViltProcessor, rows: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """为正确图文和错配图文各算一个检索分数。"""
    # 至少两张图片才能把文字错开形成负例。
    if len(rows) < 2:
        # 明确说明批量太小时为什么不能继续。
        raise ValueError("一批至少需要两张不同图片")
    # 从磁盘读取本批图片及对应描述。
    images, captions = read_batch(rows)
    # 第 i 张图改配第 i-1 句话，构造简单负例。
    wrong_captions = captions[-1:] + captions[:-1]
    # 图片复制一次：前半批配正确文字，后半批配错配文字。
    all_images = images + images
    # 文字按同样顺序拼接。
    all_captions = captions + wrong_captions
    # 处理器负责 BERT 分词、补齐文字以及 ViLT 的图片缩放与归一化。
    inputs = processor(images=all_images, text=all_captions, padding=True, truncation=True, return_tensors="pt")
    # 模型若在 GPU，输入张量也要逐个移到 GPU。
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    # ViLT 为 2B 对图文输出 2B 个原始检索分数。
    scores = model(**inputs).logits.reshape(-1)
    # 前半批是正确描述的分数。
    positive = scores[: len(rows)]
    # 后半批是错配描述的分数。
    negative = scores[len(rows) :]
    # 返回两个长度均为 B 的向量，便于逐张图片比较。
    return positive, negative


def evaluate(model: ViltForImageAndTextRetrieval, processor: ViltProcessor, rows: list[dict], batch_size: int, device: torch.device) -> tuple[float, float]:
    """计算正确描述排在错配描述之前的比例及平均分差。"""
    # eval 关闭训练行为，使前后评估可比。
    model.eval()
    # 正确排序的图片数。
    correct = 0
    # 所有图片的“正确分数减去错误分数”之和。
    margin_sum = 0.0
    # 实际参与验证的图片数。
    count = 0
    # 评估无需保留梯度，可节约显存。
    with torch.no_grad():
        # 按 batch_size 把图片清单分成数批。
        for start in range(0, len(rows), batch_size):
            # 取本批图片记录。
            batch = rows[start : start + batch_size]
            # 最后一批只剩一张时无法构造错配描述，跳过。
            if len(batch) < 2:
                continue
            # 分别得到正确与错误描述的检索分数。
            positive, negative = score_pairs(model, processor, batch, device)
            # 正确描述分数更高就记为一次排序正确。
            correct += (positive > negative).sum().item()
            # 累加所有图片的分数差。
            margin_sum += (positive - negative).sum().item()
            # 更新已评估的图片数量。
            count += len(batch)
    # 返回排序正确率和平均分差；它们不是论文报告的完整 Recall@K。
    return correct / count, margin_sum / count


def evaluate_gallery(model: ViltForImageAndTextRetrieval, processor: ViltProcessor, rows: list[dict], batch_size: int, device: torch.device) -> tuple[float, float]:
    """让每张验证图片与全部验证描述比较，计算双向 Recall@1。"""
    # 验证图片的张数也等于候选描述的条数。
    count = len(rows)
    # 固定取每张图片的第一条描述，作为该图片的正确检索目标。
    captions = [row["captions"][0] for row in rows]
    # 第 i 行、第 j 列记录“第 i 张图和第 j 句话”的分数。
    scores = torch.empty(count, count)
    # 评估模式关闭可能影响结果的训练行为。
    model.eval()
    # 评估不需要为模型保留梯度。
    with torch.no_grad():
        # 每次选一张图片作为查询。
        for image_index, row in enumerate(rows):
            # 打开这张图片并转成 RGB。
            with Image.open(DATA_ROOT / row["image"]) as source:
                # copy 让文件关闭后仍可把图片重复用于多个候选文字。
                image = source.convert("RGB").copy()
            # 全部候选描述分批送入模型，避免一次占用太多显存。
            for start in range(0, count, batch_size):
                # 当前批里要比较的几条文字。
                candidate_texts = captions[start : start + batch_size]
                # 同一张查询图片复制成与候选文字相同的数量。
                candidate_images = [image] * len(candidate_texts)
                # 处理器完成图片缩放、分词和批量补齐。
                inputs = processor(images=candidate_images, text=candidate_texts, padding=True, truncation=True, return_tensors="pt")
                # 输入张量移到模型所在设备。
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                # 检索分数移回 CPU，填入矩阵相应行和列。
                scores[image_index, start : start + len(candidate_texts)] = model(**inputs).logits.reshape(-1).cpu()
    # 每张图片选得分最高的文字；正确答案的编号等于图片编号。
    image_to_text = (scores.argmax(dim=1) == torch.arange(count)).float().mean().item()
    # 每句话选得分最高的图片；这就是反方向的文本找图。
    text_to_image = (scores.argmax(dim=0) == torch.arange(count)).float().mean().item()
    # 返回两个方向的 Recall@1。
    return image_to_text, text_to_image


def train(epochs: int, batch_size: int, lr: float) -> None:
    """先测官方权重，再冻结骨干只训练检索头，最后保存结果。"""
    # 固定随机种子，让小样本训练尽量可复查。
    torch.manual_seed(7)
    # 读取 64 张左右的训练图和独立的验证图。
    train_rows = read_rows("train")
    # 验证集不参与参数更新。
    validation_rows = read_rows("validation")
    # 检查必要的本地权重文件是否下载完成。
    if not (WEIGHT_ROOT / "pytorch_model.bin").is_file():
        # 引导读者先按说明下载官方权重。
        raise FileNotFoundError(f"缺少 ViLT 权重：{WEIGHT_ROOT / 'pytorch_model.bin'}")
    # 本地加载 ViLT 官方预处理器，不在训练时偷偷访问网络。
    processor = ViltProcessor.from_pretrained(str(WEIGHT_ROOT), local_files_only=True)
    # 本地加载 Flickr30k 微调过的 ViLT 模型。
    model = ViltForImageAndTextRetrieval.from_pretrained(str(WEIGHT_ROOT), local_files_only=True)
    # 有可用 CUDA 时使用显卡，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型参数与输入张量必须位于同一设备。
    model = model.to(device)
    # 冻结大部分参数；初学实验只训练最后的 rank_output 线性层。
    model.vilt.requires_grad_(False)
    # 只把检索头参数交给优化器，降低显存和过拟合风险。
    optimizer = torch.optim.AdamW(model.rank_output.parameters(), lr=lr)
    # 训练前先测一次，作为公开权重的基线。
    before_accuracy, before_margin = evaluate(model, processor, validation_rows, batch_size, device)
    # 在全部验证候选中测图找文与文找图的 Recall@1。
    before_i2t, before_t2i = evaluate_gallery(model, processor, validation_rows, batch_size, device)
    # 打印训练规模与基线表现。
    print(f"训练图片 {len(train_rows)} | 验证图片 {len(validation_rows)} | 设备 {device}")
    # 正确描述排在错配描述前面的比例就是这里的排序正确率。
    print(f"微调前 | 验证排序正确率 {before_accuracy:.0%} | 平均分差 {before_margin:.3f}")
    # 图找文和文找图是更严格的 16 选 1 检索任务。
    print(f"微调前 | 图找文 R@1 {before_i2t:.0%} | 文找图 R@1 {before_t2i:.0%}")
    # 每轮遍历一次小训练集。
    for epoch in range(1, epochs + 1):
        # 检索头进入训练模式。
        model.train()
        # 冻结的骨干保持评估模式。
        model.vilt.eval()
        # 记录这一轮的排序损失。
        loss_total = 0.0
        # 记录这一轮参与训练的图片数。
        count = 0
        # 按顺序逐批读取样本；第一轮先不打乱，便于复查。
        for start in range(0, len(train_rows), batch_size):
            # 取一批图片记录。
            batch = train_rows[start : start + batch_size]
            # 如果最后只有一张图，就不能产生错配文字。
            if len(batch) < 2:
                continue
            # 正确和错配描述各得到一个分数向量。
            positive, negative = score_pairs(model, processor, batch, device)
            # 当正确分数低于错误分数时，softplus 损失会较大。
            loss = F.softplus(negative - positive).mean()
            # 清空上一批的检索头梯度。
            optimizer.zero_grad()
            # 仅为检索头计算梯度。
            loss.backward()
            # 修改检索头权重，骨干参数保持原样。
            optimizer.step()
            # 把本批平均损失按图片数累加。
            loss_total += loss.item() * len(batch)
            # 更新训练图片计数。
            count += len(batch)
        # 每轮结束后用独立验证图片检查排序能力。
        val_accuracy, val_margin = evaluate(model, processor, validation_rows, batch_size, device)
        # 打印训练损失和验证指标。
        print(f"第 {epoch}/{epochs} 轮 | 训练排序损失 {loss_total / count:.3f} | 验证排序正确率 {val_accuracy:.0%} | 平均分差 {val_margin:.3f}")
    # 训练结束后再做一遍完整候选集检索。
    after_i2t, after_t2i = evaluate_gallery(model, processor, validation_rows, batch_size, device)
    # 打印两个方向的微调后 R@1。
    print(f"微调后 | 图找文 R@1 {after_i2t:.0%} | 文找图 R@1 {after_t2i:.0%}")
    # 建立保存目录。
    output_dir = ROOT / "checkpoints"
    # 如果目录已经存在就直接沿用。
    output_dir.mkdir(exist_ok=True)
    # 只保存约几 KB 的检索头，而不是复制一份 446 MB 骨干权重。
    head_path = output_dir / "vilt_pretrained_rank_head.pt"
    # state_dict 包含检索头可训练参数。
    torch.save(model.rank_output.state_dict(), head_path)
    # 把微调前后结果也保存成可直接阅读的 JSON。
    result_path = output_dir / "vilt_pretrained_metrics.json"
    # 写入评估结果与图片数量。
    result_path.write_text(json.dumps({"train_images": len(train_rows), "validation_images": len(validation_rows), "epochs": epochs, "before_pairwise_accuracy": before_accuracy, "after_pairwise_accuracy": val_accuracy, "before_image_to_text_r1": before_i2t, "before_text_to_image_r1": before_t2i, "after_image_to_text_r1": after_i2t, "after_text_to_image_r1": after_t2i, "before_margin": before_margin, "after_margin": val_margin}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 告诉读者微调后的检索头保存在哪里。
    print(f"检索头已保存：{head_path}")


if __name__ == "__main__":
    # 创建命令行选项读取器。
    parser = argparse.ArgumentParser(description="预训练 ViLT 的小数据检索头微调")
    # 默认只训练一轮，以便先比较权重前后变化。
    parser.add_argument("--epochs", type=int, default=1)
    # 每批四张图，处理器会生成八对正负图文。
    parser.add_argument("--batch-size", type=int, default=4)
    # 检索头采用较小学习率。
    parser.add_argument("--lr", type=float, default=0.0001)
    # 把命令行文字转换为 Python 参数。
    args = parser.parse_args()
    # 负例至少需要两张图，故批量不能小于 2。
    if args.batch_size < 2 or args.epochs < 1:
        # 提前说明参数范围。
        parser.error("--batch-size 至少为 2，--epochs 至少为 1")
    # 执行基线评估、微调和保存。
    train(args.epochs, args.batch_size, args.lr)
