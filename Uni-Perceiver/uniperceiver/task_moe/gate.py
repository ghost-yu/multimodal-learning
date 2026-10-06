'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# typing 标准库：Callable=可调用对象，Dict=字典类型，Tuple=元组，Optional=类型或 None。
# TYPE_CHECKING 只在静态类型检查时为 True；cast 是提示类型转换，不实际改 Tensor dtype。
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

# time 标准库提供时间；perf_counter 是高精度耗时计时器，不是训练步编号。
import time
from time import perf_counter
# torch=PyTorch，张量运算和随机分布；nn=网络组件，Tensor=类型提示。
import torch
from torch import nn
from torch import Tensor
# dist=distributed，跨进程通信；导入不等于进程组已经初始化。
import torch.distributed as dist
from torch.nn import Module, ModuleList
# F=functional；softmax 沿指定轴归一化，one_hot 将整数类别变选位向量。
import torch.nn.functional as F
# 本地事件存储用于训练日志；某些 top1 路径依赖外层已建立这个上下文。
from uniperceiver.utils.events import get_event_storage
# AMP=自动混合精度；with autocast(enabled=False) 禁止块内自动切精度，但不自动把所有输入转换到 float32。
from torch.cuda.amp import autocast



if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

# 按 device 缓存采样函数，避免重复创建分布；键没有包括 epsilon，改变噪声参数时需注意缓存行为。
uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
normal_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


import torch.distributed.nn
from uniperceiver.utils import comm
from uniperceiver.modeling.layers import FP16LayerNorm



# multiplicative=乘法，jitter=微扰；输入乘以接近 1 的随机数，不是给专家编号加噪声。
def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero,
                                                   one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


# normal=正态分布，rsample=重参数化采样；此处标准差为 1/num_expert。
# 缓存按设备建，未按 num_expert 分开；不应假设同设备动态换参数会自动刷新分布。
def normal_rsample(shape: Tuple, device: torch.device, num_expert: int) -> Tensor:
    normal = normal_map.get(device)
    if normal is None:
        std = torch.tensor(1.0/num_expert, device=device)
        mean = torch.tensor(0.0, device=device)
        normal = torch.distributions.normal.Normal(mean, std).rsample  # type: ignore
        normal_map[device] = normal
    return normal(shape)


# data 是整数编号 [N]，num_classes=E；构造 [N,E] one-hot，dtype 指返回类型。
def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes],
                         device=data.device,
                         dtype=dtype)
    # unsqueeze(-1) 将 [N] 变 [N,1]；scatter_ 按专家轴把对应位置就地写成 1。
    # 尾部下划线通常表示原地修改；这里 result 原来全是 0。
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result

@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()




class TopKGate(nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    # wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 noisy_gate_policy: Optional[str] = None,
                 cfg: dict = None,
                 moe_type: str = None,
                 **kwargs):
        super().__init__( )

        # k 是每组选择专家数，不是总专家数 E；此实现只接受 1 或 2。
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.model_dim = model_dim
        self.k = k

        self.cfg = cfg

        self.noisy_gate_policy = noisy_gate_policy
        self.noise_std = self.cfg.MOE.NOISE_STD

        self.batch_prioritized_routing = self.cfg.MOE.BATCH_PRIO
        self.gate = self.cfg.MOE.GATE_TYPE



        self.layer_type = kwargs.pop('moe_type', 'ffn')

        # tag=属性标签，transform=变换；开启后用属性向量决策，而不是直接用每个 token 内容。
        self.tag_transform_enable = self.cfg.MOE.TAG_Transform

        self.moe_type = moe_type

        if self.cfg.SOLVER.FORCE_LN_FP16:
            LayerNormModule = FP16LayerNorm
        else:
            LayerNormModule = torch.nn.LayerNorm
        if self.tag_transform_enable and self.cfg.MOE.TAG_Transform_ACT:
            self.tag_transform = torch.nn.Sequential(torch.nn.Linear(self.cfg.MOE.ATTRIBUTE_LENGTH, self.model_dim), torch.nn.GELU(),
                                                     LayerNormModule(self.model_dim))
        else:
            self.tag_transform = torch.nn.Sequential(torch.nn.Linear(self.cfg.MOE.ATTRIBUTE_LENGTH, self.model_dim), LayerNormModule(self.model_dim))

        # wg=门控权重 W_g；Linear 输入宽度 D、输出 E，参数形状 [E,D]；float 转为 float32。
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()





        pass



    # moe_embedding 通常 [N,8] 二元属性；返回 [N,D] 门控条件。
    # x 在本方法没有参与计算，不要误解释成“属性与 x 自动融合”。
    def tag_gate(self, x, data_type=None, moe_embedding:torch.Tensor = None, **kwargs):
        if self.cfg.MODEL.TAG_TRANSFORM_FP32:
            with autocast(enabled=False):
                gate_embed = self.tag_transform.float()(moe_embedding.float())
        else:
            gate_embed = self.tag_transform(moe_embedding)


        return gate_embed




    def forward(
        self,
        input,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore


        if self.tag_transform_enable:
            # 重绑定 input：从原特征变为属性条件向量；同名变量的含义和形状可能已经改变。
            input = self.tag_gate(input, **kwargs)
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        # fp32=32 位浮点；类型转换保留求导关系，不同于 detach。
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        with autocast(enabled=not self.cfg.MODEL.GATE_FP32):
            if self.cfg.SOLVER.FORCE_WG_RECAST:
                # used for dbeugging only
                logits = self.wg.half().float()(input_fp32)
            else:
                logits = self.wg(input_fp32)

        # 注意原实现此分支计算 gate_output 后没有 return；return 缩进在 else 内。
        # 这个静态边界尚未修复，不要把 k=1 的全部配置当作已验证可用。
        if self.k == 1 and self.gate == 'deepspeed':
            gate_output = self.top1gating(
                logits,
                self.noisy_gate_policy if self.training else None,
                **kwargs)

        # tutel gate function
        else:
            gate_output = self.top2gating(
                logits,
                self.noisy_gate_policy if self.training else None,
                **kwargs )


            return gate_output

    def load_balance(self, gates, mask1, num_experts, data_type=None):
        # Compute l_aux
        if self.balance_loss and self.training:
            # TODO: for retrieval task, these maybe some gpu do not have  this input

            if data_type == 'INPUT':
                if comm._LOCAL_IMAGE_LENGTH > 0 and not comm._LOCAL_UTOKEN_LENGTH + comm._LOCAL_GTOKEN_LENGTH > 0:
                    # input image features only
                    me = gates.sum(dim=0)
                    ce = mask1.sum(dim=0)

                    # maybe has retrieval pair
                    if comm._MOE_TARGET_MECE_LIST.get(str(comm._LOCAL_CURRENT_LAYER)+'_'+self.layer_type, None) is not None:
                        # if len(comm._MOE_TARGET_MECE_LIST) > 0:
                        me_t, ce_t = comm._MOE_TARGET_MECE_LIST[
                            str(comm._LOCAL_CURRENT_LAYER) + '_' +
                            self.layer_type]
                        me = me + me_t
                        ce = ce + ce_t

                    me = me * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    ce = ce * self.task_weights[comm._LOCAL_CURRENT_TASK]

                elif comm._LOCAL_IMAGE_LENGTH > 0 and comm._LOCAL_UTOKEN_LENGTH + comm._LOCAL_GTOKEN_LENGTH > 0:

                    # sum of these two distribution from two modalities
                    me = gates.sum(dim=0)
                    ce = mask1.sum(dim=0)

                    me = me * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    ce = ce * self.task_weights[comm._LOCAL_CURRENT_TASK]

                elif comm._LOCAL_IMAGE_LENGTH <= 0 and comm._LOCAL_UTOKEN_LENGTH + comm._LOCAL_GTOKEN_LENGTH > 0:

                    me = gates.sum(
                        dim=0) * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    ce = mask1.sum(
                        dim=0) * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    # raise NotImplementedError
                else:

                    raise NotImplementedError

            elif data_type == 'TARGET':
                # the retrieval embedding

                # only remove the padding contributions

                comm._MOE_TARGET_MECE_LIST[str(comm._LOCAL_CURRENT_LAYER) + '_' +self.layer_type] = [gates.sum(dim=0), mask1.sum(dim=0)]

            elif data_type == 'IN_LABEL':
                # remove paddings contributions

                me = gates.sum(dim=0)
                ce = mask1.sum(dim=0)

            elif data_type == 'WORD_VOCAB':
                # do not need padding mask
                me = gates.sum(dim=0)
                ce = mask1.sum(dim=0)
            else:
                raise NotImplementedError

            # debug left

            if not data_type == 'TARGET':
                me = torch.distributed.nn.all_reduce(
                    me) / comm.get_world_size()
                ce = torch.distributed.nn.all_reduce(
                    ce) / comm.get_world_size()

                if data_type not in comm._MOE_LOSSES_COLLECTIONS[
                        'exp_balance']:
                    comm._MOE_LOSSES_COLLECTIONS['exp_balance'][
                        data_type] = []
                comm._MOE_LOSSES_COLLECTIONS['exp_balance'][
                    data_type].append([me, ce])


    def top1gating(
            self,
            logits: Tensor,
            noisy_gate_policy: Optional[str] = None,
            **kwargs,
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implements Top1Gating on logits."""

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device)
        elif noisy_gate_policy == 'vmoe':
            # logits 为 [N,E]；最后轴长度 E 是专家总数，N 是路由条件组数量。
            num_experts = int(logits.shape[-1])
            logits_w_noise = logits + normal_rsample(logits.shape,
                                                     device=logits.device,
                                                     num_expert=num_experts/self.noise_std)

        # everything is in fp32 in this function
        gates = F.softmax(logits, dim=1)
        # Create a mask for 1st's expert per token
        # noisy gating
        indices1_s = torch.argmax(logits_w_noise if logits_w_noise is not None else gates, dim=1)

        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        # gating decisions
        exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

        self.load_balance(gates, mask1, num_experts)

        self.tb_output(
            mask1,
            exp_counts,
            gates=None
        )

        gates = (gates*mask1).sum(dim=1)
        self.tb_output(mask1=None, exp_counts=None, gates=[gates])

        return [indices1_s], [gates]




    def top2gating(
        self,
        logits: Tensor,
        noisy_gate_policy: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implements Top2Gating on logits."""
        # everything is in fp32 in this function

        # logits 为 [N,E]；最后轴长度 E 是专家总数，N 是路由条件组数量。
        num_experts = int(logits.shape[-1])

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device) * self.noise_std
        elif noisy_gate_policy == 'vmoe':
            logits_w_noise = logits + normal_rsample(logits.shape,
                                                     device=logits.device,
                                                     num_expert=num_experts/self.noise_std)

        # topk_indices = torch.topk(logits, self.k, dim=1).indices
        topk_indices = torch.topk(
            logits_w_noise
            if logits_w_noise is not None else logits,
            self.k,
            dim=1).indices

        # topk_indices=[N,k]；chunk 沿列拆 k 份，再 view(-1) 得 k 个 [N] 下标 Tensor。
        # s 是本地后缀，表示多组下标；它不表示额外的序列轴。
        indices_s = [x.view(-1) for x in topk_indices.chunk(self.k, dim=1)]
        masks_se = [
            one_hot_with_dtype(x, num_classes=num_experts, dtype=x.dtype)
            for x in indices_s
        ]


        if noisy_gate_policy == 'vmoe':
            gates = F.softmax(logits_w_noise, dim=1)

        else:
            gates = F.softmax(logits, dim=1)

        # self.load_balance(gates, masks_se[0], num_experts)
        # one-hot mask 从 [N,E] softmax 概率里取出各选中专家权重，得到 k 个 [N]。
        # 此处 x 是 mask 循环变量，不是之前的 token 特征 x。
        gates_s = [(gates * x).sum(dim=1) for x in masks_se]

        # gating decisions
        exp_counts = torch.sum(masks_se[0], dim=0).detach().to('cpu')

        # self.tb_output(masks_se[0], exp_counts, gates=None)
        # if self.k>1:
        #     for k in range(1, self.k):
        #         self.tb_output(masks_se[k], torch.sum(masks_se[k], dim=0).detach().to('cpu'), None, postfix='_top{}'.format(k+1))



        if self.k > 1:

            # Normalize Gate
            # denom=denominator（分母）；对选中权重求和并用机器 epsilon 防除零。
            denom_s = torch.clamp(sum(gates_s),
                                  min=torch.finfo(gates_s[0].dtype).eps)
            # 对选中权重重新归一化，k>1 时每组总权重为 1；未选中专家不会参加融合。
            gates_s = [x / denom_s for x in gates_s]

        # self.tb_output(mask1=None, exp_counts=None, gates=gates_s)

        # 返回二元 tuple（下标列表,权重列表）；旧类型提示列四项，不能只读提示推断返回结构。
        return indices_s, gates_s


    def tb_output(self, data_type=None, mask1=None, exp_counts=None, gates=None, postfix=''):
        if self.training:
            storage = get_event_storage()
        else:
            return

        if not (comm._LOCAL_CURRENT_TASK == 'imagenet' or comm._LOCAL_CURRENT_TASK.startswith('bookswiki') or comm._LOCAL_CURRENT_TASK.startswith('cc3m') or comm._LOCAL_CURRENT_TASK.startswith('cc12m') or comm._LOCAL_CURRENT_TASK.startswith('tqa')):
            # to save time
            return

        if  (storage._iter+1)%(comm._EXPERT_LOG_INTERVAL//10) != 0:
            # to save time
            return


        if storage is not None and comm.is_main_process():
            # pass
            # for each expert

            if gates is not None:
                if data_type == "INPUT" and comm._LOCAL_IMAGE_LENGTH > 0:


                    gate_logs = {
                        "logits_layer{}_expert_{}/top{}_{}_{}_v".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type,
                            e_id+1, comm._LOCAL_CURRENT_TASK,
                            data_type): ratio[0]
                        for e_id, ratio in enumerate(gates)
                    }
                    storage.put_scalars(**gate_logs, avg_hint=True)


                    if gates[0].shape[0] > 1:
                        gates_t_logs = {
                            "logits_layer{}_expert_{}/top{}_{}_{}_t".
                            format(comm._LOCAL_CURRENT_LAYER,
                                    self.layer_type, e_id+1,
                                    comm._LOCAL_CURRENT_TASK,
                                    data_type): ratio[1]
                            for e_id, ratio in enumerate(gates)
                        }
                        storage.put_scalars(**gates_t_logs, avg_hint=True)

                elif data_type in ['IN_LABEL', 'WORD_VOCAB']:

                    gates_logs = {
                        "logits_layer{}_expert_{}/top{}_{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type,
                            e_id+1, data_type): ratio[0]
                        for e_id, ratio in enumerate(gates)
                    }
                    storage.put_scalars(**gates_logs, avg_hint=True)

                else:

                    gates_logs = {
                        "layer{}_expert_{}/top{}_{}_{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type,
                            e_id+1, comm._LOCAL_CURRENT_TASK,
                            data_type): ratio[0]
                        for e_id, ratio in enumerate(gates)
                    }
                    storage.put_scalars(**gates_logs, avg_hint=True)

            else:

                if data_type == "INPUT" and comm._LOCAL_IMAGE_LENGTH > 0:

                    exp_counts_v = mask1[0]
                    exp_count_logs = {
                        "layer{}_expert_{}/E{}_{}_{}_v{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                            comm._LOCAL_CURRENT_TASK, data_type,
                            postfix): ratio
                        for e_id, ratio in enumerate((exp_counts_v /
                                                    exp_counts_v.sum()).tolist())
                    }
                    storage.put_scalars(**exp_count_logs, avg_hint=True)

                    if mask1.size(0)>1:
                        exp_counts_t = mask1[1]
                        exp_count_logs = {
                            "layer{}_expert_{}/E{}_{}_{}_t{}".format(
                                comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                                comm._LOCAL_CURRENT_TASK,
                                data_type, postfix): ratio
                            for e_id, ratio in enumerate((
                                exp_counts_t / exp_counts_t.sum()).tolist())
                        }
                        storage.put_scalars(**exp_count_logs, avg_hint=True)



                elif data_type in ['IN_LABEL', 'WORD_VOCAB']:
                    exp_count_logs = {
                        "layer{}_expert_{}/E{}_{}{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                            data_type, postfix): ratio
                        for e_id, ratio in enumerate((exp_counts /
                                                    exp_counts.sum()).tolist())
                    }
                    storage.put_scalars(**exp_count_logs, avg_hint=True)

                else:
                    exp_count_logs = {
                        "layer{}_expert_{}/E{}_{}_{}{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                            comm._LOCAL_CURRENT_TASK, data_type,
                            postfix): ratio
                        for e_id, ratio in enumerate((exp_counts /
                                                    exp_counts.sum()).tolist())
                    }
                    storage.put_scalars(**exp_count_logs, avg_hint=True)
