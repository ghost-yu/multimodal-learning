"""加载刚训练的小 CLIP，查看某张验证图片最像哪些文字。"""

# argparse 读取要看的验证图片编号。
import argparse
# Path 定位保存的权重和图片清单。
from pathlib import Path

# PyTorch 负责恢复参数和计算检索分数。
import torch

# 模型结构必须与训练权重保存时完全相同。
from clip_tiny import TinyCLIP
# 复用训练时的图片缩放和单词转 ID 规则。
from flickr8k_data import Flickr8kMini, read_manifest

# 当前脚本所在的“小型复现”文件夹。
ROOT = Path(__file__).resolve().parent
# 已下载的 Flickr8k 小样本所在位置。
DATA_ROOT = ROOT / "real_data" / "flickr8k-mini"
# 训练脚本输出的 CLIP 权重文件。
CHECKPOINT = ROOT / "checkpoints" / "clip_flickr8k_mini.pt"


def main(query_index: int) -> None:
    """打印一张图的前三名描述，以及其正确描述找出的前三张图。"""
    # weights_only=True 只读取张量和简单数据结构。
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    # 用训练时的词表和模型尺寸，避免输入编号或参数形状不匹配。
    model = TinyCLIP(**saved["model_args"])
    # 把训练好的参数写入同样结构的空模型。
    model.load_state_dict(saved["model"])
    # 有显卡时用显卡；这个小模型用 CPU 也能推理。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型和数据必须位于同一设备。
    model = model.to(device)
    # eval 关闭训练行为。
    model.eval()
    # 验证清单含图片路径及其五条原始描述。
    rows = read_manifest(DATA_ROOT, "validation")
    # 查询编号必须落在现有验证集范围之内。
    if query_index < 0 or query_index >= len(rows):
        # 直接说明合法范围，方便修改 --index。
        raise ValueError(f"--index 必须在 0 到 {len(rows) - 1} 之间")
    # 固定每张图的第一条描述，与训练脚本验证方式一致。
    dataset = Flickr8kMini(DATA_ROOT, rows, saved["vocab"], saved["model_args"]["text_len"], saved["model_args"]["image_size"], training=False)
    # 每张验证图片的像素张量先放入列表。
    images = []
    # 每条对应描述的整数 ID 也放入列表。
    texts = []
    # 依序读取，确保图片 i 与句子 i 仍是一对。
    for index in range(len(dataset)):
        # Dataset 按训练时相同方式缩图和分词。
        image, text = dataset[index]
        # 添加一张图的 [3,64,64] 张量。
        images.append(image)
        # 添加一句话的 [20] 整数张量。
        texts.append(text)
    # stack 把 16 张图合成 [16,3,64,64]，然后移到模型设备。
    image_batch = torch.stack(images).to(device)
    # 16 条文字合成 [16,20]，位置次序与图片保持一致。
    text_batch = torch.stack(texts).to(device)
    # 推理不需要梯度，可以减少显存。
    with torch.no_grad():
        # 每张图独立编码为一个单位长度向量。
        image_vectors = model.encode_image(image_batch)
        # 每句话也独立编码为一个单位长度向量。
        text_vectors = model.encode_text(text_batch)
        # 一次矩阵乘法形成 [16,16] 余弦相似度表。
        scores = image_vectors @ text_vectors.T
    # 当前图片在验证清单中的路径。
    print(f"查询图片：{DATA_ROOT / rows[query_index]['image']}")
    # 清单中与这张图对应的第一条正确描述。
    print(f"对应描述：{rows[query_index]['captions'][0]}")
    # 第 query_index 行包含这张图与 16 条描述的相似度。
    top_texts = scores[query_index].topk(min(3, len(rows))).indices.tolist()
    # 列出图找文排名前三的描述。
    print("图找文前三名：")
    # enumerate 生成从 1 开始的名次编号。
    for rank, text_index in enumerate(top_texts, start=1):
        # 相似度是归一化向量点积，不是概率。
        score = scores[query_index, text_index].item()
        # 同时标明该描述是否真的来自查询图片。
        print(f"  {rank}. 候选 {text_index:02d} | 相似度 {score:.3f} | {'正确' if text_index == query_index else '其他图片'} | {rows[text_index]['captions'][0]}")
    # 第 query_index 列对应正确描述与 16 张图片的相似度。
    top_images = scores[:, query_index].topk(min(3, len(rows))).indices.tolist()
    # 列出文找图排名前三的图片文件。
    print("文找图前三名：")
    # 依照相似度从高到低逐个打印。
    for rank, image_index in enumerate(top_images, start=1):
        # 读取该图文组合的相似度。
        score = scores[image_index, query_index].item()
        # 打印候选图片和正确与否。
        print(f"  {rank}. 候选 {image_index:02d} | 相似度 {score:.3f} | {'正确' if image_index == query_index else '其他图片'} | {rows[image_index]['image']}")


if __name__ == "__main__":
    # 建立命令行参数读取器。
    parser = argparse.ArgumentParser(description="查看 CLIP 教学模型的具体检索结果")
    # 默认检查验证集第 0 张图，也可指定 0 到 15 的其他编号。
    parser.add_argument("--index", type=int, default=0)
    # 解析用户输入。
    args = parser.parse_args()
    # 加载权重并显示排名。
    main(args.index)
