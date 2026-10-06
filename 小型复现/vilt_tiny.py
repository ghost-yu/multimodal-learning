"""ViLT 教学版：图文 token 直接拼接，再做匹配分类。"""

# argparse 用来读取命令行里的 --steps 等选项。
import argparse
# PyTorch 提供张量与自动求导。
import torch
# nn 里有卷积、词向量、Transformer 和线性层。
from torch import nn
# F 提供交叉熵等常用损失函数。
from torch.nn import functional as F

# 从同目录的数据文件导入造图、造词和配对函数。
from data import VOCAB, make_pairs


class TinyViLT(nn.Module):
    """只保留 ViLT 的图文早期融合与 ITM（图文匹配）任务。"""

    def __init__(self, dim: int = 32, image_size: int = 16, patch_size: int = 4, text_len: int = 4, vocab_size: int = len(VOCAB)):
        # 初始化 nn.Module，PyTorch 才能追踪所有可训练参数。
        super().__init__()
        # patch_size 控制每块的宽高；教学默认每块 4×4，真实图可用 8×8。
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        # 词表大小可切换；编号 0 作为 PAD，填充向量保持为零。
        self.word_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        # CLS 是整个图文对的可学习汇总 token。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # 按图片尺寸计算有多少 patch；默认 16×16 图片产生 16 个。
        patch_count = (image_size // patch_size) ** 2
        # 总长度由 1 个 CLS、text_len 个文字 token 和 patch_count 个图片 token 组成。
        self.position = nn.Parameter(torch.randn(1, 1 + text_len + patch_count, dim) * 0.02)
        # 类型 0 表示 CLS/文字，类型 1 表示图片块。
        self.type_embed = nn.Embedding(2, dim)
        # 一层小 Transformer；batch_first=True 表示输入形状为 [B,序列长,D]。
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True)
        # 包一层 Encoder 就得到共享的图文编码器。
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        # CLS 的输出被映射成“不匹配/匹配”两个分数。
        self.classifier = nn.Linear(dim, 2)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """输入图片 [B,3,H,W] 和文字 [B,T]；输出 [B,2]。"""
        # 从图片读出批量大小 B。
        batch_size = images.shape[0]
        # 卷积后形状 [B,D,网格高,网格宽]。
        patches = self.patch_embed(images)
        # 将图片网格展平，再换轴成 [B,图片块数,D]。
        patches = patches.flatten(2).transpose(1, 2)
        # 词编号通过查表变成 [B,文字长度,D] 的词向量。
        words = self.word_embed(texts)
        # 原 CLS 参数是 [1,1,D]；扩成 [B,1,D] 供每个样本使用。
        cls = self.cls_token.expand(batch_size, -1, -1)
        # dim=1 是序列维；默认玩具数据得到 [B,21,D]。
        tokens = torch.cat((cls, words, patches), dim=1)
        # 前 1+T 个位置属于 CLS/文字，后面属于图片。
        type_ids = torch.cat((torch.zeros(1 + words.shape[1], dtype=torch.long, device=images.device), torch.ones(patches.shape[1], dtype=torch.long, device=images.device)))
        # 加上位置信息和模态类型信息；广播会把它们用于整个 batch。
        tokens = tokens + self.position + self.type_embed(type_ids).unsqueeze(0)
        # CLS 和图片没有填充，因此它们的掩码全为 False。
        visible_mask = torch.zeros(batch_size, 1 + patches.shape[1], dtype=torch.bool, device=images.device)
        # 文字编号 0 是 PAD；True 告诉 Transformer 忽略该位置。
        padding_mask = torch.cat((visible_mask[:, :1], texts.eq(0), visible_mask[:, 1:]), dim=1)
        # 所有图文 token 一起经过共享 Transformer，实现早期融合。
        mixed = self.encoder(tokens, src_key_padding_mask=padding_mask)
        # 取第 0 个位置的 CLS 向量，得到 [B,D]。
        pooled = mixed[:, 0]
        # 线性分类层输出 [B,2]，交叉熵会把它当作两个类别的原始分数。
        return self.classifier(pooled)


def train(steps: int, lr: float) -> None:
    """重复生成小数据并训练，最后打印图文匹配预测。"""
    # 固定随机种子，尽量让多次运行的初始化一致。
    torch.manual_seed(7)
    # 有 CUDA 时用显卡，否则用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型，并把参数移动到与数据相同的设备。
    model = TinyViLT().to(device)
    # AdamW 会根据梯度更新模型参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 切换到训练模式；本模型没有 dropout，保留这一步以符合标准写法。
    model.train()
    # 从第 1 步循环到指定步数。
    for step in range(1, steps + 1):
        # 每一步都造 4 对正确图文和 4 对错误图文。
        images, texts, labels = make_pairs(device)
        # 前向计算得到 8 条样本各自的两个分类分数。
        logits = model(images, texts)
        # 交叉熵让正确类别的分数变高。
        loss = F.cross_entropy(logits, labels)
        # 清空上一步留下的参数梯度。
        optimizer.zero_grad()
        # 自动求导，计算损失对每个参数的梯度。
        loss.backward()
        # 按梯度实际修改参数。
        optimizer.step()
        # 每约五分之一进度打印一次，避免输出过长。
        if step == 1 or step % max(1, steps // 5) == 0:
            # argmax 取两个类别分数较大的那个类别。
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            # 打印当前损失和这批训练数据上的准确率。
            print(f"ViLT 步数 {step:3d}/{steps} | 损失 {loss.item():.4f} | 本批准确率 {accuracy:.0%}")
    # 训练结束后切换到评估模式。
    model.eval()
    # 评估不需要计算梯度，可省内存和时间。
    with torch.no_grad():
        # 再生成一批图片、正确文字及错误文字。
        images, texts, labels = make_pairs(device)
        # 给每个图文对计算分类分数。
        logits = model(images, texts)
        # 取“匹配”类别的概率，方便初学者阅读。
        probabilities = logits.softmax(dim=1)[:, 1]
        # 统计这一批样本的预测正确比例。
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        # 显示每一条样本的真实标签和匹配概率。
        for index in range(labels.numel()):
            # 文本中第二、三个 token 对应颜色和形状。
            color_id, shape_id = texts[index, 1:3].tolist()
            # 根据两个 token 的编号还原成人能读的短句。
            text_name = ("红色" if color_id == VOCAB["RED"] else "蓝色") + ("方形" if shape_id == VOCAB["SQUARE"] else "圆形")
            # 同一张图片在前后两个半批中各出现一次。
            image_index = index % 4
            # 用这批里的编号标记图片；真实类别由图像像素本身体现。
            image_name = "示例图片" + str(image_index + 1)
            # 输出标签和模型认为“匹配”的概率。
            print(f"{image_name} + {text_name} | 标签 {labels[index].item()} | 匹配概率 {probabilities[index].item():.3f}")
        # 单独打印最终准确率，方便比较不同步数。
        print(f"ViLT 最终本批准确率：{accuracy:.0%}")


if __name__ == "__main__":
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="ViLT 教学版：图文匹配")
    # --steps 控制训练迭代次数。
    parser.add_argument("--steps", type=int, default=120)
    # --lr 控制学习率，即每次参数更新的幅度。
    parser.add_argument("--lr", type=float, default=0.003)
    # 把命令行文字转换成 Python 参数对象。
    args = parser.parse_args()
    # 正式调用训练函数。
    train(args.steps, args.lr)
