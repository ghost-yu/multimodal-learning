'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

# PyTorch 管理参数、Tensor 与求导；这里实现专家投影，不是任务损失。
import torch
# copy 是标准库；deepcopy 递归复制专家模板，使各专家有独立 Parameter。
import copy
# 本地 one-hot 工具：把整数编号变成选位向量；本文件该 import 不意味着 forward 一定调用它。
from .gate import one_hot_with_dtype
# comm=communication，但这里也保存 caption 生成模式的项目级状态。
from uniperceiver.utils import comm

# F=functional；F.linear(x,W,b)=x @ W.T + b，W 形状 [Dout,Din]。
import torch.nn.functional as F

# autocast 是 CUDA 混合精度上下文工具；不能仅凭 import 推断本文件所有操作都启用 AMP。
from torch.cuda.amp import autocast


# Fused=融合；Experts=专家集合。对同一条件的一组 token，融合线性权重来节省重复乘法。
class FusedExperts(torch.nn.Module):
    def __init__(self, expert, cfg,  num_local_experts=1):
        super(FusedExperts, self).__init__()
        self.cfg = cfg

        # 名称沿用 DeepSpeed，但实际存为本地 ModuleList；名称不证明已经采用跨设备专家并行。
        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # bias=偏置；根据首个专家是否有 bias 决定是否一起加权偏置。当前实现假设模板兼容这个属性。
        self.bias_merge = self.deepspeed_experts[0].bias is not None


    # indice 是 index 的非标准命名，记录各条件组专家编号；gate 是相应权重。
    # 条件组数为 1/2，不是 B 或 L；当前实现只支持一个或两个模态组。
    def top1_expert_forward(self, x, indice, gate, mode=None, **kwargs):
        assert  mode is None, "unified qkv inference is not supported for top1"
        if indice.size(0)== 1:
            #unimodal
            x = self.deepspeed_experts[indice[0]](x) * gate[0].to(x)
        elif indice.size(0) == 2:
            # mulmodal
            # data_cum_length=cumulative length（累计长度）；第 1 个边界用于沿序列切成两组。
            # 这里 x[:, :边界, :] 和 x[:, 边界:, :] 各用各自路由，不是每个 token 任意选择一组。
            data1_length = kwargs['sample_info']['data_cum_length'][1]
            x = torch.cat([
                self.deepspeed_experts[indice[0]](x[:, :data1_length, :]) * gate[0].to(x),
                self.deepspeed_experts[indice[1]](x[:, data1_length:, :]) * gate[1].to(x)
            ],
                          dim=1)

        else:
            raise NotImplementedError('only support one or two modality')
        return x

    # merge=合并；先计算 Wmix=gate1*W1+gate2*W2，bmix 同理，再执行一次 F.linear。
    # 只有线性映射及组内一致门控才能这样重排；不能把完整含非线性的两个 MLP 任意合并。
    def mergelayer(self, x,  index1, index2, gate1, gate2, mode=None):
        
        if not self.cfg.SOLVER.FORCE_EXPERT_ADDING_FP16:
            # q=query；把合并 QKV 参数的输出行切出前 1/3，用于 query 投影。
            if mode == 'q':
                # hidden_states
                _start = 0
                # Linear 权重第 0 维是输出宽度；假设按 Q/K/V 三段存放，所以 //3 得一段输出宽度。
                _end = self.deepspeed_experts[index1].weight.shape[0] // 3
                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:_end, :] * gate1 +
                    self.deepspeed_experts[index2].weight[_start:_end, :] * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:_end] * gate1 +
                    self.deepspeed_experts[index2].bias[_start:_end] * gate2
                    if self.bias_merge else None,
                )

            # kv=key/value；取后 2/3 输出行，配合历史特征生成 key 和 value。
            elif mode == 'kv':
                # history_states
                _start =  self.deepspeed_experts[index1].weight.shape[0] // 3

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:, :] * gate1 +
                    self.deepspeed_experts[index2].weight[_start:, :] * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:] * gate1 +
                    self.deepspeed_experts[index2].bias[_start:] * gate2
                    if self.bias_merge else None,
                )

            else:

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight * gate1 +
                    self.deepspeed_experts[index2].weight * gate2,
                    bias=self.deepspeed_experts[index1].bias * gate1 +
                    self.deepspeed_experts[index2].bias * gate2 if self.bias_merge else None,
                )
        else:
            # q=query；把合并 QKV 参数的输出行切出前 1/3，用于 query 投影。
            if mode == 'q':
                # hidden_states
                _start = 0
                # Linear 权重第 0 维是输出宽度；假设按 Q/K/V 三段存放，所以 //3 得一段输出宽度。
                _end = self.deepspeed_experts[index1].weight.shape[0] // 3
                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:_end, :].half() * gate1 +
                    self.deepspeed_experts[index2].weight[_start:_end, :].half() * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:_end].half() * gate1 +
                    self.deepspeed_experts[index2].bias[_start:_end].half() * gate2 if self.bias_merge else None,
                )

            # kv=key/value；取后 2/3 输出行，配合历史特征生成 key 和 value。
            elif mode == 'kv':
                # history_states
                _start = self.deepspeed_experts[index1].weight.shape[0] // 3

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:, :].half() * gate1 +
                    self.deepspeed_experts[index2].weight[_start:, :].half() * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:].half() * gate1 +
                    self.deepspeed_experts[index2].bias[_start:].half() * gate2 if self.bias_merge else None,
                )

            else:

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight.half() * gate1 + self.deepspeed_experts[index2].weight.half() * gate2,
                    bias=self.deepspeed_experts[index1].bias.half() * gate1 +
                    self.deepspeed_experts[index2].bias.half() * gate2 if self.bias_merge else None,
                )
        

    # top2 激活两名专家，indices/gates 各是长度为 2 的列表，每项按条件组给编号/权重。
    def top2_expert_forward(self, x, indices, gates, mode=None, **kwargs):

        # caption eval mode
        # caption 生成单步时序列长度为 1；源码特殊选取第 2 个条件组，应结合生成调用检查属性对齐。
        if comm._CAPTION_GEN_MODE and x.shape[1] == 1:
            #
            return self.mergelayer(x,
                                   indices[0][1], indices[1][1],
                                   gates[0][1], gates[1][1], mode=mode)

        # unimodal
        if indices[0].size(0) == 1:
            x = self.mergelayer(x, indices[0][0], indices[1][0], gates[0][0], gates[1][0], mode=mode)
        elif indices[0].size(0) == 2:
            # data_cum_length=cumulative length（累计长度）；第 1 个边界用于沿序列切成两组。
            # 这里 x[:, :边界, :] 和 x[:, 边界:, :] 各用各自路由，不是每个 token 任意选择一组。
            data1_length = kwargs['sample_info']['data_cum_length'][1]
            if mode == 'kv' and kwargs['sample_info'].get('pe_length', 0) > 0:
                # may have prompt embedding for kv embedding
                data1_length += kwargs['sample_info'].get('pe_length', 0)
            x = torch.cat([
                self.mergelayer(x[:, :data1_length, :], indices[0][0], indices[1][0], gates[0][0], gates[1][0], mode=mode),
                self.mergelayer(x[:, data1_length:, :], indices[0][1], indices[1][1], gates[0][1], gates[1][1], mode=mode)
            ],
                          dim=1)

        else:
            raise NotImplementedError('only support one or two modality')
        return x

    def forward(self, hidden_states, top_indices=None, gates=None, **kwargs):

        # top1
        if len(top_indices) == 1:
            out = self.top1_expert_forward(hidden_states, top_indices[0], gates[0], **kwargs)

        # top2
        elif len(top_indices) == 2:
            out = self.top2_expert_forward(hidden_states, top_indices, gates, **kwargs)

        else:
            raise NotImplementedError("only support top1 and top2 ")



        # assert 检查序列长度没有改变；只检查这一维，不能据此保证所有数值和其他形状正确。
        assert out.shape[1] == hidden_states.shape[1]

        return out
