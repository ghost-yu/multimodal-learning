# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# copy 是 Python 标准库；deepcopy 递归复制模块，使 A/B 参数成为不同对象，不共享同一张权重。
import copy

# torch 提供 split/cat 张量分割和拼接；nn.Module 使子模块参数能被 optimizer 与 state_dict 找到。
import torch
import torch.nn as nn


def MultiwayWrapper(args, module, dim=1):
    # Wrapper=包装器工厂；multiway 开关启用时，把一个普通模块变为 A/B 两分支模块。
    # dim=1 是 [B,L,D] 的序列轴，按 token 来源切分；不是按 batch 分两组。
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    # 返回一个闭包函数 apply_fn，它会记住 position，稍后由 nn.Module.apply 遍历子模块时调用。
    def apply_fn(module):
        # hasattr 判断对象是否有这个属性；没有路由属性的普通注意力层不受影响。
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    # Network=网络；Multiway=多路。这里的路由由模态边界决定，没有可学习的 top-k gate。
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        # A 接收原模块，BEiT3 中对应视觉；B 对应文字。字母是源码标识，不是性能等级。
        self.A = module
        # 深复制使两路权重独立；与 VLMo 等方法的“共享自注意力、模态专用 FFN”思路相近。
        self.B = copy.deepcopy(module)
        # reset_parameters 重新初始化 B，避免它只是永久使用与 A 一样的初始参数。
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        # -1=全视觉，0=全文字，正数=前 split_position 个视觉 token、其余文字 token。
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        # x=[B,L,D]；split 指定两个段长，返回 x1 与 x2；dim 为序列轴时保留 batch 与通道。
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        # x1/x2 是两模态输入，y1/y2 是各自专家输出；**kwargs 原样转发额外关键字参数。
        # 两路 FFN 对每个 token 处理，不在这里计算图文注意力；共享注意力发生在 EncoderLayer。
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        # cat 按原序恢复视觉在前、文字在后，不打乱样本；输出序列长度与输入一致。
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    # 保留源代码 Mutliway 拼写；这类直接接收两张已经构造好的嵌入表，不再 deepcopy 普通模块。
    def __init__(self, modules, dim=1):
        # 显式 super(类,self) 从 MultiwayNetwork 的父类开始调用，即 nn.Module 初始化。
        # 它跳过 MultiwayNetwork.__init__，因为本类已有两张不同大小的位置表，不需要复制并 reset。
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1
