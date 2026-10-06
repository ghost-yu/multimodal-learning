"""用同一批 Flickr8k 验证图比较两个从零训练的小模型。"""

# argparse 允许只看 ViLT、只看 ALBEF，或两者都看。
import argparse
# Path 定位数据与之前保存的权重。
from pathlib import Path

# PyTorch 用来加载权重、计算张量与检索矩阵。
import torch
# ALBEF 教学版模型结构。
from albef_tiny import TinyALBEF
# 真实数据集读取器和清单读取函数。
from flickr8k_data import Flickr8kMini, read_manifest
# ViLT 教学版模型结构。
from vilt_tiny import TinyViLT

# 脚本所在目录就是项目根目录。
ROOT = Path(__file__).resolve().parent
# 下载的图片与描述保存于此。
DATA_ROOT = ROOT / "real_data" / "flickr8k-mini"
# 前一步训练生成的权重保存于此。
CHECKPOINT_ROOT = ROOT / "checkpoints"


def load_data(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """用 checkpoint 内的词表把验证数据转成固定形状张量。"""
    # 验证清单独立于训练清单。
    rows = read_manifest(DATA_ROOT, "validation")
    # 模型创建时用的尺寸配置保存在 checkpoint 里。
    settings = checkpoint["model_args"]
    # validation=False 表示固定使用每张图第一条文字。
    dataset = Flickr8kMini(DATA_ROOT, rows, checkpoint["vocab"], settings["text_len"], settings["image_size"], training=False)
    # 把所有图片张量放到一个列表。
    images = []
    # 把每张图对应的文字张量放到另一个列表。
    texts = []
    # 按编号读取验证数据集。
    for index in range(len(dataset)):
        # 一次取出一张图片与它正确的描述。
        image, text = dataset[index]
        # 保存图片以备后面形成所有图文组合。
        images.append(image)
        # 保存对应的文字 token。
        texts.append(text)
    # stack 将多个 [3,H,W] 拼成 [N,3,H,W]。
    image_batch = torch.stack(images)
    # 文字也堆成 [N,T]。
    text_batch = torch.stack(texts)
    # 返回一一对应的图片和文字。
    return image_batch, text_batch


def make_score_matrix(name: str, model: torch.nn.Module, images: torch.Tensor, texts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """生成 [N,N] 矩阵：行是图片，列是候选描述。"""
    # N 是验证图片数量。
    count = images.shape[0]
    # 分数矩阵放在 CPU，方便最后直接打印指标。
    scores = torch.empty(count, count)
    # 评估模式不更新模型参数。
    model.eval()
    # 评估期间不保存梯度，减少显存占用。
    with torch.no_grad():
        # ALBEF 先分别编码所有图片与文字，避免重复计算。
        if name == "albef":
            # 所有图片的 patch 特征形状 [N,P,D]。
            image_features = model.encode_image(images.to(device))
            # 所有文字的 token 特征形状 [N,T,D]。
            text_features = model.encode_text(texts.to(device))
        # 每一行对应一张查询图片。
        for image_index in range(count):
            # ViLT 每一对图文都需要一起送进共享 Transformer。
            if name == "vilt":
                # 将第 i 张图片复制 N 份，与 N 条候选描述逐一配对。
                repeated_images = images[image_index : image_index + 1].expand(count, -1, -1, -1).to(device)
                # 模型给每一对输出“不匹配/匹配”两个分数。
                logits = model(repeated_images, texts.to(device))
            else:
                # ALBEF 的图像特征已算好，只需复制当前图像的特征。
                repeated_features = image_features[image_index : image_index + 1].expand(count, -1, -1)
                # 让 N 条文字分别读取这一张图片的 patch。
                logits = model.matching_logits(repeated_features, text_features)
            # “匹配分数减去不匹配分数”可用于候选描述排序。
            scores[image_index] = (logits[:, 1] - logits[:, 0]).cpu()
    # 返回整张图文匹配分数表。
    return scores


def evaluate(name: str) -> None:
    """读取一个模型权重并打印图找文、文找图 Recall@1。"""
    # 文件名与真实训练入口保存权重时一致。
    checkpoint_path = CHECKPOINT_ROOT / f"{name}_flickr8k_mini.pt"
    # weights_only=True 只接受张量等安全类型。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 用 checkpoint 的词表和尺寸读取相同验证集。
    images, texts = load_data(checkpoint)
    # 选择和保存权重对应的模型类。
    model_class = TinyViLT if name == "vilt" else TinyALBEF
    # 用训练时保存的配置重新创建同一结构。
    model = model_class(**checkpoint["model_args"])
    # 把训练后的参数填入该结构。
    model.load_state_dict(checkpoint["model"])
    # 有空闲显卡时用显卡，否则用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型搬到计算设备。
    model = model.to(device)
    # 计算每张图对每条文字的分数。
    scores = make_score_matrix(name, model, images, texts, device)
    # 真实答案是对角线，即第 i 张图对应第 i 句话。
    targets = torch.arange(scores.shape[0])
    # 每张图挑分数最高的文字，得到图找文 Recall@1。
    image_to_text = (scores.argmax(dim=1) == targets).float().mean().item()
    # 每句话挑分数最高的图片，得到文找图 Recall@1。
    text_to_image = (scores.argmax(dim=0) == targets).float().mean().item()
    # 打印候选数量，让读者知道 R@1 是几选一。
    print(f"{name}: {scores.shape[0]} 个候选 | 图找文 R@1 {image_to_text:.0%} | 文找图 R@1 {text_to_image:.0%}")


if __name__ == "__main__":
    # 定义命令行参数。
    parser = argparse.ArgumentParser(description="比较两个从零训练的小模型的检索效果")
    # 默认 both 会依次评估两个模型。
    parser.add_argument("--model", choices=("vilt", "albef", "both"), default="both")
    # 把命令行文字变成参数值。
    args = parser.parse_args()
    # both 时先评估 ViLT，再评估 ALBEF。
    names = ("vilt", "albef") if args.model == "both" else (args.model,)
    # 依次打印每个模型的验证检索结果。
    for name in names:
        # 调用完整加载与评估函数。
        evaluate(name)
