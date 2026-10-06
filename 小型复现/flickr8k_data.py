"""把本地 Flickr8k 小样本转成两份教学模型可读取的张量。"""

# Counter 统计训练文字中每个英文词出现的次数。
from collections import Counter
# json 读取下载脚本生成的图片和描述清单。
import json
# Path 统一处理 Windows 与其他系统的路径。
from pathlib import Path
# random 每次训练可以选同一张图的不同描述。
import random
# re 用简单规则把英文句子切成单词。
import re

# NumPy 把 Pillow 图片高效地变成数字数组。
import numpy as np
# Pillow 负责打开图片和保持长宽比地缩放。
from PIL import Image, ImageOps
# PyTorch 把图片和文字数组变成模型输入张量。
import torch
# Dataset 是 PyTorch 数据集的标准接口。
from torch.utils.data import Dataset

# 这四个特殊 token 的编号固定，不随数据集改变。
SPECIAL = {"PAD": 0, "UNK": 1, "CLS": 2, "SEP": 3}


def split_words(sentence: str) -> list[str]:
    """将英文描述转换成小写单词列表。"""
    # lower 统一大小写；正则只保留连续字母或数字。
    return re.findall(r"[a-z]+|[0-9]+", sentence.lower())


def read_manifest(data_root: Path, split: str) -> list[dict]:
    """读取 train.json 或 validation.json，并检查文件是否存在。"""
    # 训练与验证分别对应不同的 JSON 文件。
    path = data_root / f"{split}.json"
    # 若未下载数据，给出能直接操作的提示。
    if not path.is_file():
        # FileNotFoundError 会让入口脚本停止，不会错误地空跑。
        raise FileNotFoundError(f"找不到 {path}；请先运行 下载Flickr8k样本.py")
    # 用 UTF-8 读取英文描述，兼容将来可能加入的中文。
    with path.open("r", encoding="utf-8") as handle:
        # JSON 顶层是由多张图片记录组成的列表。
        rows = json.load(handle)
    # 空清单无法训练，提前报错比稍后出现奇怪张量错误更好。
    if not rows:
        # 明确指出是哪个文件没有数据。
        raise ValueError(f"{path} 没有图文样本")
    # 把所有行交给词表构造或 Dataset。
    return rows


def build_vocab(train_rows: list[dict], max_words: int = 2000) -> dict[str, int]:
    """仅根据训练集建立词表，避免提前偷看验证集。"""
    # Counter 会累加每个英文词出现的次数。
    counts = Counter()
    # 每张图片有五条描述，都可用于收集训练词汇。
    for row in train_rows:
        # 遍历该图片对应的五条文字。
        for caption in row["captions"]:
            # 将句子分词，并把词频加入计数器。
            counts.update(split_words(caption))
    # 先拷贝 PAD、UNK、CLS、SEP 四个特殊 token。
    vocab = dict(SPECIAL)
    # 从高频到低频取最多 max_words 个真实单词。
    for word, _ in counts.most_common(max_words):
        # 不覆盖已经分配的特殊 token。
        if word not in vocab:
            # 新词编号就是当前词表的长度。
            vocab[word] = len(vocab)
    # 返回“单词 → 整数编号”字典。
    return vocab


class Flickr8kMini(Dataset):
    """每次返回一张 64×64 图片和一条长度固定的文字。"""

    def __init__(self, data_root: Path, rows: list[dict], vocab: dict[str, int], text_len: int = 20, image_size: int = 64, training: bool = False):
        # 保存数据集根目录，用于拼出图片完整路径。
        self.data_root = data_root
        # 保存图文记录列表。
        self.rows = rows
        # 保存由训练集构造的词表。
        self.vocab = vocab
        # 所有句子都整理成相同长度，才能合成一个 batch。
        self.text_len = text_len
        # 图片统一缩放到这一尺寸。
        self.image_size = image_size
        # 训练时随机选一条描述；验证时固定用第一条。
        self.training = training

    def __len__(self) -> int:
        # DataLoader 通过这个值知道有多少张图片。
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据编号取出一张图片的文件名和五条描述。
        row = self.rows[index]
        # 将相对图片路径拼成磁盘上的完整路径。
        image_path = self.data_root / row["image"]
        # 用 with 打开图片，使用完后自动关闭文件。
        with Image.open(image_path) as original:
            # pad 保持原图比例，不够方的部分补黑边。
            image = ImageOps.pad(original.convert("RGB"), (self.image_size, self.image_size))
            # 先转换成 [H,W,3] 的 float32 数组，再把像素缩放到 0～1。
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        # PyTorch 卷积层要求通道在前，得到 [3,H,W]。
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        # 训练时随机选五条描述之一，增加文字变化。
        caption = random.choice(row["captions"]) if self.training else row["captions"][0]
        # 最多保留 text_len-2 个真实单词，给 CLS 与 SEP 留两个位置。
        words = split_words(caption)[: self.text_len - 2]
        # 句首放 CLS，模型用它汇总整句话。
        token_ids = [SPECIAL["CLS"]]
        # 已知词取词表编号，未见过的词取 UNK 编号。
        token_ids += [self.vocab.get(word, SPECIAL["UNK"]) for word in words]
        # 句尾放 SEP，表示文字到此结束。
        token_ids.append(SPECIAL["SEP"])
        # 句子不足 text_len 时，在末尾补 PAD=0。
        token_ids += [SPECIAL["PAD"]] * (self.text_len - len(token_ids))
        # 返回图片张量与整数类型的文字张量。
        return image_tensor, torch.tensor(token_ids, dtype=torch.long)
