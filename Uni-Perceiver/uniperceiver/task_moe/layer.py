'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch.nn.init as init
import torch
from torch import nn
import torch.distributed as dist




# gate=门控/路由器；TopK 表示选得分最高的 k 个专家，当前仅支持 k=1/2。
from .gate import TopKGate
import copy
import typing

# Fused=融合；as Experts 为本地类别名；这里能将线性专家权重先加权，再只做一次投影。
from .experts import FusedExperts as Experts


# Task=任务；MoE=Mixture of Experts（专家混合）。此模块组合 gate 与多个专家。
class TaskMoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 cfg=None):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.

            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).

            num_experts (int, optional): default=1, the total number of experts per layer.

            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.

            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.

            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.

            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.

            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.

            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).

            use_rts (bool, optional): default=True, whether to use Random Token Selection.

            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        """

        super().__init__()


        # num=数量，experts=专家；总专家数 E 不等于每次选中的 k。
        self.num_experts = num_experts

        # 检查传入专家模板类型；当前支持线性或注意力声明，并非任意 MLP 都能在后续 FusedExperts 中使用。
        if isinstance(expert, nn.Linear):
            self.expert_type = 'linear'
        elif isinstance(expert, nn.MultiheadAttention):
            self.expert_type = 'attention'
        else:
            raise NotImplementedError('please check expert type')

        # 从模板复制 E 份参数；不是 E 个名字指向同一个模块。
        experts = Experts(expert, cfg, num_experts)

        self.gate = TopKGate(hidden_size,
                             num_experts,
                             k,
                             noisy_gate_policy,
                             cfg,
                             moe_type=self.expert_type)


        self.experts = experts



    # hidden_states 是特征，通常 [B,L,D]；gate_decision 为可复用的 [专家下标列表, 权重列表]。
    # kwargs=可选命名参数，例如 sample_info、moe_embedding；需要与 gate/experts 的约定一致。
    def forward(self, hidden_states, gate_decision=None, **kwargs):
        """ MoE forward
        Arguments:
            hidden_states (Tensor): input to the layer
        Returns:
            A tuple including output
            * output (Tensor): output of the model
        """


        # 已有路由时不重新决策，使相关线性投影复用同样选择；None 才调用 gate。
        if  gate_decision is not None:
            top_indices, gates = gate_decision
        else:
            top_indices, gates = self.gate(hidden_states, **kwargs)

        # top_indices 给出选中的专家编号；gates 给出各专家贡献权重，不是输入 token 的有效性 mask。
        expert_output = self.experts(hidden_states, top_indices, gates, **kwargs)

        # 返回二元 tuple：结果及可供后续模块复用的路由决定；不是只返回一个 Tensor。
        return expert_output, [top_indices, gates]
