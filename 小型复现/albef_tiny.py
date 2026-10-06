"""ALBEF 教学版：先用 ITC 对齐图文，再用 ITM 融合图文。"""

# argparse 读取命令行参数。
import argparse
# PyTorch 负责张量、自动求导和设备管理。
import torch
# nn 包含卷积、词向量、注意力和分类层。
from torch import nn
# F 包含交叉熵与向量归一化函数。
from torch.nn import functional as F

# 共用图片、文字和配对数据生成函数。
from data import CLASS_NAMES, VOCAB, make_images, make_text


class TinyALBEF(nn.Module):
    """仅实现 ALBEF 的图文对齐 ITC 与图文融合 ITM。"""

    def __init__(self, dim: int = 32, temperature: float = 0.1, image_size: int = 16, patch_size: int = 4, text_len: int = 4, vocab_size: int = len(VOCAB)):
        # 让 nn.Module 正确登记下面定义的参数。
        super().__init__()
        # patch_size 控制图片每块像素宽高；默认 4×4。
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        # 根据图片尺寸计算 patch 数量。
        patch_count = (image_size // patch_size) ** 2
        # 图片位置向量告诉编码器 patch 原来在图片的哪里。
        self.image_position = nn.Parameter(torch.randn(1, patch_count, dim) * 0.02)
        # 文字 token 编号变为 dim 维向量；0 留给 PAD。
        self.word_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        # 每个文字位置分别有自己的位置向量。
        self.text_position = nn.Parameter(torch.randn(1, text_len, dim) * 0.02)
        # 图片编码器的一层 Transformer，输入 [B,16,D]。
        image_layer = nn.TransformerEncoderLayer(dim, 4, 64, dropout=0.0, batch_first=True)
        # 保留所有 patch 的输出，后面做图文融合还要使用它们。
        self.image_encoder = nn.TransformerEncoder(image_layer, 1, enable_nested_tensor=False)
        # 文字编码器的另一层 Transformer，输入 [B,4,D]。
        text_layer = nn.TransformerEncoderLayer(dim, 4, 64, dropout=0.0, batch_first=True)
        # 图像与文字先各自编码，这就是 ALBEF 的“先对齐”部分。
        self.text_encoder = nn.TransformerEncoder(text_layer, 1, enable_nested_tensor=False)
        # 文字做 Query、图片做 Key/Value，让词读取相关图像块。
        self.cross_attention = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        # 融合后的文字特征经此层得到“不匹配/匹配”分数。
        self.matching_head = nn.Linear(dim, 2)
        # 温度缩放图文相似度；数值越小，softmax 区分越明显。
        self.temperature = temperature

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """图片 [B,3,H,W] → 图片块特征 [B,图片块数,D]。"""
        # 卷积把图片切成网格，输出 [B,D,网格高,网格宽]。
        patches = self.patch_embed(images)
        # 展平网格并换轴，得到 [B,16,D]。
        patches = patches.flatten(2).transpose(1, 2)
        # 加上位置向量，再让各 patch 相互交换信息。
        return self.image_encoder(patches + self.image_position)

    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """文字 [B,T] → 各词特征 [B,T,D]。"""
        # token 编号查词向量表，再加句中位置向量。
        words = self.word_embed(texts) + self.text_position
        # 让文字 token 通过自注意力互相读取信息；0 是需要忽略的 PAD。
        return self.text_encoder(words, src_key_padding_mask=texts.eq(0))

    def contrastive_loss(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """ITC：正确图文在相似度矩阵的对角线上。"""
        # 对 16 个 patch 求平均，得到整张图片向量 [B,D]。
        image_global = image_tokens.mean(dim=1)
        # 取文本首个 CLS token 的输出，得到整句向量 [B,D]。
        text_global = text_tokens[:, 0]
        # 向量除以自身长度，避免只靠放大数值提高相似度。
        image_global = F.normalize(image_global, dim=-1)
        # 文字向量也用同样方法归一化。
        text_global = F.normalize(text_global, dim=-1)
        # 矩阵乘法得到 [B,B]：第 i 张图对第 j 句话的相似度。
        similarity = image_global @ text_global.T / self.temperature
        # 本批四类图文按相同次序排列，所以正确答案是对角线 0、1、2、3。
        target = torch.arange(similarity.shape[0], device=similarity.device)
        # 行方向：每张图应选中同编号的文字。
        image_to_text = F.cross_entropy(similarity, target)
        # 列方向：每句话应选中同编号的图片。
        text_to_image = F.cross_entropy(similarity.T, target)
        # 两个方向平均，即对称图文对比损失。
        return (image_to_text + text_to_image) / 2

    def matching_logits(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """ITM：通过交叉注意力判断一对图文是否匹配。"""
        # Query 是文字；Key 和 Value 都是图片 patch。
        attended, _ = self.cross_attention(text_tokens, image_tokens, image_tokens, need_weights=False)
        # 残差连接保留原文字信息，再加入图像信息。
        fused = text_tokens + attended
        # 取融合后的 CLS token，得到每对图文的整体特征。
        pooled = fused[:, 0]
        # 输出两个类别的原始分数 [B,2]。
        return self.matching_head(pooled)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入四对正确图文；返回 ITC、ITM 以及 ITM 的分类分数。"""
        # 图片和文字分别编码，得到两组 token 特征。
        image_tokens = self.encode_image(images)
        # 文字编码器尚未看到图片。
        text_tokens = self.encode_text(texts)
        # 用整图与整句向量计算“先对齐”的 ITC 损失。
        itc_loss = self.contrastive_loss(image_tokens, text_tokens)
        # 将文字样本错开一位，构造错误配对的文字特征。
        negative_text_tokens = text_tokens.roll(shifts=1, dims=0)
        # 正例与负例的图像 token 都来自同一批图片。
        all_image_tokens = torch.cat((image_tokens, image_tokens), dim=0)
        # 前半批是正确文字，后半批是错配文字。
        all_text_tokens = torch.cat((text_tokens, negative_text_tokens), dim=0)
        # 所有 2B 个图文对分别经过交叉注意力和匹配分类器。
        itm_logits = self.matching_logits(all_image_tokens, all_text_tokens)
        # 前 B 个标签为 1，后 B 个标签为 0。
        labels = torch.cat((torch.ones(images.shape[0], dtype=torch.long, device=images.device), torch.zeros(images.shape[0], dtype=torch.long, device=images.device)))
        # 交叉熵训练分类器分辨正确配对和错误配对。
        itm_loss = F.cross_entropy(itm_logits, labels)
        # 将两个损失与预测一并交给训练函数。
        return itc_loss, itm_loss, itm_logits


def train(steps: int, lr: float) -> None:
    """训练两条目标，并用新生成的图片检查匹配分类。"""
    # 固定随机种子，让实验更容易复查。
    torch.manual_seed(7)
    # 自动选择 CUDA 显卡或 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型并移至计算设备。
    model = TinyALBEF().to(device)
    # 优化器根据两个损失的梯度更新模型参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 进入训练模式。
    model.train()
    # 重复进行指定次数的训练。
    for step in range(1, steps + 1):
        # 每一步随机打乱四个类别，保证一批中每类恰好出现一次。
        class_ids = torch.randperm(4)
        # 根据类别生成四张图片并移到模型所在设备。
        images = make_images(class_ids).to(device)
        # 根据相同类别生成四条正确文字并移到同一设备。
        texts = make_text(class_ids).to(device)
        # 前向同时得到 ITC、ITM 两种损失及匹配分类结果。
        itc_loss, itm_loss, itm_logits = model(images, texts)
        # 两种任务等权相加，让共享编码器同时学习对齐与融合。
        total_loss = itc_loss + itm_loss
        # 清空上次迭代残留的梯度。
        optimizer.zero_grad()
        # 自动计算每个参数对总损失的梯度。
        total_loss.backward()
        # 依据梯度更新模型。
        optimizer.step()
        # 每约五分之一进度输出一次训练情况。
        if step == 1 or step % max(1, steps // 5) == 0:
            # ITM 前半批为正例、后半批为负例。
            labels = torch.cat((torch.ones(4, dtype=torch.long, device=device), torch.zeros(4, dtype=torch.long, device=device)))
            # 比较分类结果与真实标签，求平均准确率。
            accuracy = (itm_logits.argmax(dim=1) == labels).float().mean().item()
            # 分开显示两个损失，便于看清不同目标的训练变化。
            print(f"ALBEF 步数 {step:3d}/{steps} | ITC {itc_loss.item():.4f} | ITM {itm_loss.item():.4f} | 本批准确率 {accuracy:.0%}")
    # 测试时关闭训练模式。
    model.eval()
    # 不保存梯度，减少评估时的额外开销。
    with torch.no_grad():
        # 构造一批新的四类图片。
        class_ids = torch.randperm(4)
        # 把图片移到同一计算设备。
        images = make_images(class_ids).to(device)
        # 正确的文字与图片类别一一对应。
        texts = make_text(class_ids).to(device)
        # 模型返回 ITC、ITM 损失和 8 对图文的匹配分数。
        _, _, logits = model(images, texts)
        # 前四个应匹配，后四个应不匹配。
        labels = torch.cat((torch.ones(4, dtype=torch.long, device=device), torch.zeros(4, dtype=torch.long, device=device)))
        # 用 softmax 把原始分数变为“匹配”概率。
        probabilities = logits.softmax(dim=1)[:, 1]
        # 计算最终这一批新样本的正确比例。
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        # 逐条打印，让你看出正负例的概率区别。
        for index in range(labels.numel()):
            # 前半批的文字类别等于图片类别。
            image_class = class_ids[index % 4].item()
            # 后半批的文字来自前一个位置，因为模型使用 roll(shifts=1)。
            text_class = class_ids[index % 4].item() if index < 4 else class_ids[(index % 4 - 1) % 4].item()
            # 用中文名称展示图片类别、文字类别及预测。
            print(f"图片 {CLASS_NAMES[image_class]} + 文字 {CLASS_NAMES[text_class]} | 标签 {labels[index].item()} | 匹配概率 {probabilities[index].item():.3f}")
        # 输出最终一批数据的准确率。
        print(f"ALBEF 最终本批准确率：{accuracy:.0%}")


if __name__ == "__main__":
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="ALBEF 教学版：先对齐，再融合")
    # --steps 指定训练多少次。
    parser.add_argument("--steps", type=int, default=250)
    # --lr 指定优化器每次更新的大致步幅。
    parser.add_argument("--lr", type=float, default=0.001)
    # 解析用户在命令行输入的参数。
    args = parser.parse_args()
    # 开始训练与评估。
    train(args.steps, args.lr)
