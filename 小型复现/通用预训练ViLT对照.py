"""用只做过通用图文预训练的 ViLT 检查 Flickr8k 小样本检索。"""

# argparse 读取训练轮数和每批样本数。
import argparse
# json 把最后的评估数字保存为可阅读的文本。
import json

# torch 提供加载权重、梯度和优化器。
import torch
# softplus 用于图文正负配对的排序损失。
from torch.nn import functional as F
# 复用已经逐行解释过的官方 ViLT 预处理与检索结构。
from transformers import ViltForImageAndTextRetrieval, ViltProcessor

# 这些函数负责数据读取、二选一排序验证与 16 选 1 完整检索。
from 预训练ViLT小微调 import ROOT, WEIGHT_ROOT, evaluate, evaluate_gallery, read_rows, score_pairs

# 通用预训练权重存放在另一个目录，不能与 Flickr30k 微调权重混淆。
BASE_ROOT = ROOT / "pretrained" / "vilt-mlm-itm"


def load_base_model(device: torch.device) -> ViltForImageAndTextRetrieval:
    """把通用预训练的二分类 ITM 头接到检索模型上。"""
    # 使用 Transformers 兼容层读取 ViLT 大骨干；此时检索头还未从权重初始化。
    model = ViltForImageAndTextRetrieval.from_pretrained(str(BASE_ROOT), local_files_only=True)
    # 单独读取原 checkpoint，找出预训练时用的 image-text matching 分类头。
    state = torch.load(BASE_ROOT / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # 原 ITM 头输出“不匹配、匹配”两个分数；正类通常是第 1 行。
    weight_name = "itm_score.fc.weight"
    # 配套的偏置向量也有两项。
    bias_name = "itm_score.fc.bias"
    # 如果 checkpoint 没有这两个键，就不能假装用了已训练的检索头。
    if weight_name not in state or bias_name not in state:
        # 列出实际带 itm_score 的键，帮助排查模型版本差异。
        names = [name for name in state if "itm_score" in name]
        # 报错后停止，绝不悄悄使用随机初始化的头部。
        raise KeyError(f"找不到预训练 ITM 权重；现有相关键：{names}")
    # 预训练 ITM 头应有两行，每行对应 768 维 pooled 特征。
    if state[weight_name].shape != (2, model.config.hidden_size):
        # 形状不对通常意味着用了不兼容的 checkpoint。
        raise ValueError(f"ITM 权重形状异常：{tuple(state[weight_name].shape)}")
    # 修改参数数值时不需要计算梯度。
    with torch.no_grad():
        # 论文检索实验把预训练 ITM 的“真配对”输出行作为检索分数头的初值。
        model.rank_output.weight.copy_(state[weight_name][1:2])
        # 同样复制“真配对”输出行的偏置。
        model.rank_output.bias.copy_(state[bias_name][1:2])
    # 只在头部复制完成之后才把模型移动到 GPU 或 CPU。
    return model.to(device)


def main(epochs: int, batch_size: int, lr: float) -> None:
    """测预训练基线、训练小检索头并再次测量。"""
    # 固定参数初始化等随机过程。
    torch.manual_seed(7)
    # 本地的 64 张图片只用于更新检索头。
    train_rows = read_rows("train")
    # 另外 16 张图片只用于验证，不参与梯度更新。
    validation_rows = read_rows("validation")
    # 无需重复下载处理器；两个 checkpoint 都使用 ViLT 配套的 BERT 分词和图片预处理。
    processor = ViltProcessor.from_pretrained(str(WEIGHT_ROOT), local_files_only=True)
    # GPU 可用时用 GPU，否则回退到 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取通用图文预训练骨干，并从 ITM 头初始化检索头。
    model = load_base_model(device)
    # 冻结大骨干，这台电脑只需为最后的线性层计算梯度。
    model.vilt.requires_grad_(False)
    # 优化器只会修改 rank_output 两个参数张量。
    optimizer = torch.optim.AdamW(model.rank_output.parameters(), lr=lr)
    # 二选一验证：一条正确描述和一条错配描述比较。
    before_accuracy, before_margin = evaluate(model, processor, validation_rows, batch_size, device)
    # 完整验证：每张图在 16 条描述中找自己，反向也一样。
    before_i2t, before_t2i = evaluate_gallery(model, processor, validation_rows, batch_size, device)
    # 打印微调前的表现。
    print(f"通用预训练 ViLT | 训练图 {len(train_rows)} | 验证图 {len(validation_rows)} | {device}")
    print(f"微调前 | 二选一 {before_accuracy:.0%} | 分差 {before_margin:.3f} | 图找文 R@1 {before_i2t:.0%} | 文找图 R@1 {before_t2i:.0%}")
    # 每轮完整遍历 64 张训练图。
    for epoch in range(1, epochs + 1):
        # 训练模式只用于检索头；骨干被冻住且保持评估模式。
        model.train()
        model.vilt.eval()
        # 逐批积累用于打印的排序损失。
        loss_total = 0.0
        # 记录参与训练的图片数。
        count = 0
        # 从训练清单中依次取 batch_size 张图。
        for start in range(0, len(train_rows), batch_size):
            # 当前批的记录。
            batch = train_rows[start : start + batch_size]
            # 至少两张图片才能把文字错位造负例。
            if len(batch) < 2:
                continue
            # 同一张图分别配正确文字和错误文字，得到两组分数。
            positive, negative = score_pairs(model, processor, batch, device)
            # 只有错误分数过高时才付出明显损失。
            loss = F.softplus(negative - positive).mean()
            # 清掉上一步遗留的梯度。
            optimizer.zero_grad()
            # 求当前损失对检索头参数的导数。
            loss.backward()
            # 让正确图文的分数相对提高。
            optimizer.step()
            # 累加本批损失，后面按图片数求平均。
            loss_total += loss.item() * len(batch)
            # 累加图片数量。
            count += len(batch)
        # 每轮结束后重新计算二选一指标。
        after_accuracy, after_margin = evaluate(model, processor, validation_rows, batch_size, device)
        # 打印这一轮是否有训练收益。
        print(f"第 {epoch}/{epochs} 轮 | 训练损失 {loss_total / count:.3f} | 验证二选一 {after_accuracy:.0%} | 分差 {after_margin:.3f}")
    # 全部轮次完成后计算较严格的双向 16 选 1 检索。
    after_i2t, after_t2i = evaluate_gallery(model, processor, validation_rows, batch_size, device)
    # 打印最终的双向检索结果。
    print(f"微调后 | 图找文 R@1 {after_i2t:.0%} | 文找图 R@1 {after_t2i:.0%}")
    # 保存结果与检索头到 checkpoints 文件夹。
    output_dir = ROOT / "checkpoints"
    # 已存在时不报错。
    output_dir.mkdir(exist_ok=True)
    # 只保存很小的检索头；要再次推理仍需要原通用预训练权重。
    torch.save(model.rank_output.state_dict(), output_dir / "vilt_base_rank_head.pt")
    # JSON 中标明使用了哪种预训练权重，防止与 Flickr30k 权重的结果混淆。
    result = {"checkpoint": "dandelin/vilt-b32-mlm-itm", "train_images": len(train_rows), "validation_images": len(validation_rows), "epochs": epochs, "before_pairwise_accuracy": before_accuracy, "after_pairwise_accuracy": after_accuracy, "before_image_to_text_r1": before_i2t, "before_text_to_image_r1": before_t2i, "after_image_to_text_r1": after_i2t, "after_text_to_image_r1": after_t2i, "before_margin": before_margin, "after_margin": after_margin}
    # 生成带缩进的文本，供初学者直接打开查看。
    (output_dir / "vilt_base_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # 初始化命令行参数读取器。
    parser = argparse.ArgumentParser(description="通用预训练 ViLT 对照实验")
    # 默认只训练一轮，先观察小样本是否带来增益。
    parser.add_argument("--epochs", type=int, default=1)
    # 每批四张图片对应四条正例和四条负例。
    parser.add_argument("--batch-size", type=int, default=4)
    # 检索头参数很少，使用小学习率。
    parser.add_argument("--lr", type=float, default=0.0001)
    # 解析用户提供的参数。
    args = parser.parse_args()
    # 至少两张图才有负例；轮数也须为正。
    if args.batch_size < 2 or args.epochs < 1:
        # 清晰解释输入范围。
        parser.error("--batch-size 至少为 2，--epochs 至少为 1")
    # 运行完整的基线、训练与复测流程。
    main(args.epochs, args.batch_size, args.lr)
