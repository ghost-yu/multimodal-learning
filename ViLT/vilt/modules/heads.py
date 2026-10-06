# torch 建张量，例如 MLMHead 中 torch.zeros 初始化词表偏置。
import torch
# nn = neural network；Module 是可训练层基类，Linear 做线性映射。
import torch.nn as nn
# F 是函数式算子常见别名；原仓库在此文件导入但没有实际调用。
import torch.nn.functional as F

# BERT 提供的 PredictionHeadTransform 是“线性层 + 激活 + 归一化”式预测前处理。
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    # pooler 把整段融合序列的第 0 个文字 CLS 位置压成图文对的整句向量。
    def __init__(self, hidden_size):
        # hidden_size=D，即每个 token 的特征维数；输入/输出宽度都为 D。
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Tanh 把数值压入 (-1,1)，不改变 [B,D] 形状。
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states 是共同 Transformer 的 [B,T+P,D]；[:,0] 取序列第 0 位。
        first_token_tensor = hidden_states[:, 0]
        # dense 是有可训练权重的全连接层；pooled_output 仍为 [B,D]。
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    # ITM = image-text matching，判断图和文是否原本配对。
    def __init__(self, hidden_size):
        super().__init__()
        # 2 是“假配对/真配对”两类，输出 raw logits，不是概率。
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x 为 Pooler 输出的 [B,D]；fc 后变 [B,2]。
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    # MLM = masked language modeling；每个文字位置预测 V 个词的分数。
    def __init__(self, config, weight=None):
        super().__init__()
        # config 中含 hidden_size=D、vocab_size=V；transform 保持 [B,T,D]。
        self.transform = BertPredictionHeadTransform(config)
        # decoder 将 D 维文字特征投到整个词表 V，输出 [B,T,V]。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Parameter 明确表示可训练参数；词表每个词有一项偏置。
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            # 可选与词嵌入矩阵共享权重；当前主模型调用时没有传 weight。
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    # MPP = masked patch prediction；每个 patch 的 RGB 三通道各分 256 档。
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # 256*3 个输出可重排为 [B,P,3,256]，逐颜色通道预测 0..255。
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
