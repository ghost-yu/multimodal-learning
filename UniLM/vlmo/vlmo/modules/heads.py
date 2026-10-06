# torch 创建 MLM 词表偏置；nn 提供全部任务头所需的线性层。
import torch
import torch.nn as nn
import torch.nn.functional as F

# BERT 的词预测变换包含线性层、激活函数和归一化。
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    """把序列第 0 个 CLS token 变为整条图文序列的向量。"""
    def __init__(self, hidden_size):
        # 初始化模块并登记后续参数。
        super().__init__()
        # [B,D] → [B,D] 的线性变换。
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Tanh 把输出压到 -1 和 1 之间。
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states 为 [B,N,D]；第 0 个位置是 CLS。
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    """ITM 二分类：输出图文不匹配/匹配两个分数。"""
    def __init__(self, hidden_size):
        super().__init__()
        # 最后一维 2 对应两个类别，尚未经过 softmax。
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

class ITCHead(nn.Module):
    """ITC 向量投影：为图文点积准备同维度表示。"""
    def __init__(self, hidden_size):
        super().__init__()
        # 无偏置线性层保持特征维 D 不变，参数按模态分别拥有。
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    """MLM 词预测：每个文字位置输出整个词表的分数。"""
    def __init__(self, config, weight=None):
        super().__init__()
        # 先对隐藏特征做 BERT 风格变换。
        self.transform = BertPredictionHeadTransform(config)
        # [B,T,D] → [B,T,V]，V 是词表大小。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 每个词表项有独立可学习偏置。
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            # 可选：与已有的词嵌入矩阵共享解码权重。
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x
