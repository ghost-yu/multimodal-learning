# 张量运算由 PyTorch 完成；Metric 帮我们跨 batch 累计统计量。
import torch
from torchmetrics import Metric


class Accuracy(Metric):
    """普通分类准确率：预测正确的数量 / 有效标签数量。"""
    def __init__(self, dist_sync_on_step=False):
        # 先初始化 torchmetrics 的父类，使状态能被框架管理。
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # add_state 注册累计量；多 GPU 汇总时对各进程求和。
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        # detach 使统计过程不参与反向传播，并把两个张量移到计数器设备。
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        # 在最后一维找分数最高的类别；MLM 时最后一维是词表大小。
        preds = logits.argmax(dim=-1)
        # -100 是“忽略该位置”的标签，只统计真正需要预测的位置。
        preds = preds[target != -100]
        target = target[target != -100]
        # 整个 batch 没有有效目标时，保持累计状态不变。
        if target.numel() == 0:
            return 1

        # 预测和标签应一一对应，否则上游形状有误。
        assert preds.shape == target.shape

        # 每个布尔值 True 代表预测正确，求和后加入历史正确数。
        self.correct += torch.sum(preds == target)
        # numel() 返回有效标签的元素个数。
        self.total += target.numel()

    def compute(self):
        # epoch 末统一计算，而不是简单平均每个 batch 的准确率。
        return self.correct / self.total


class Scalar(Metric):
    """记录一个标量（通常是 loss）的 batch 平均值。"""
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # scalar 存累计和，total 存累计更新次数。
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        # 已是张量就移到同一设备；Python 数字先转成 float 张量。
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        # 每调用一次 update，就累计一个 batch 的数值。
        self.scalar += scalar
        self.total += 1

    def compute(self):
        # 这里按 batch 次数平均；batch 大小不同也仍然等权。
        return self.scalar / self.total


class VQAScore(Metric):
    """VQA 常用软分数：预测答案获得目标分数，而非只判对错。"""
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # score 是总得分，total 是样本数。
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        # 计算指标不需要梯度；float 便于后面的乘法。
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        # 每张图的问题选分数最高的答案类别。
        logits = torch.max(logits, 1)[1]
        # 建一个与目标分数矩阵同形状、全为 0 的 one-hot 容器。
        one_hots = torch.zeros(*target.size()).to(target)
        # scatter_ 在每行预测类别的位置填 1；下划线表示原地修改。
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        # 只留下预测答案对应的人工软标签分数。
        scores = one_hots * target

        # 把本批答案得分累加，len(logits) 是本批样本数。
        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        # 每个问题的平均得分即为 VQA 指标。
        return self.score / self.total
