# 本脚本检查原 MoEAttentionBlock 的稠密数学路径和原 Encoder 层的配置边界。
# 属性专家使用接口替身检查门控传参；不等于运行真实 TaskMoE/全模型训练。
import ast  # AST=抽象语法树，从本地文件提取原类，避开未安装的分布式/融合依赖。
import json  # JSON标准库，记录真实检查结果。
import math  # math.sqrt提供原类默认引用的平方根。
from pathlib import Path  # pathlib标准路径库，定位项目根目录和输出路径。
from types import SimpleNamespace as NS  # NS为便于阅读的别名，模拟cfg.MOE等层级配置属性。
from typing import Optional, Union, Callable  # 原类签名引用的类型提示，不参与计算。
from functools import partial  # 原类用partial绑定TaskMoE构造参数。
import torch  # PyTorch，实际执行原类张量计算，当前使用CPU。
from torch import nn, Tensor  # 神经网络模块和Tensor类型，供原类定义使用。
import torch.nn.functional as F  # 原注意力的linear/softmax，原FFN默认relu。

root = Path(__file__).resolve().parents[2]  # scripts→复现→Uni-Perceiver项目目录。
source = root / 'uniperceiver/modeling/layers/transformer_encoder_moe_layer.py'  # 注释过的原文件。
tree = ast.parse(source.read_text(encoding='utf-8'))  # 读取并解析，不自动执行顶部可选依赖导入。
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ['MoETransformerEncoderLayer', 'MoEAttentionBlock']]  # 选择两个真实类。


class TraceTaskMoE(nn.Module):  # Trace=记录调用；此替身不选择/融合多专家，只检查原层如何传gate。
    calls = []  # 保存各次调用的mode与gate对象标识。

    def __init__(self, hidden_size, expert, **kwargs):  # 形参匹配partial(TaskMoE,公共配置)后的调用。
        super().__init__()  # 初始化Module，下面expert能注册为子模块。
        self.expert = expert  # 存储原线性层，实现形状保持与可用参数。

    def forward(self, inputs, gate_decision=None, mode=None, **kwargs):  # 返回(线性结果,gate)，匹配实际TaskMoE接口。
        if gate_decision is None:  # 尚无可复用门控时，要求调用者传属性。
            gate_decision = {'attribute': kwargs['moe_embedding']}  # 缺属性则KeyError，便于暴露Post-LN未转发kwargs的行为。
        self.calls.append({'mode': mode, 'gate_id': id(gate_decision)})  # gate_id只用于进程内检查复用身份，不是跨运行稳定编号。
        weight, bias = self.expert.weight, self.expert.bias  # 原Linear参数；这里没有建立专家副本。
        if mode == 'q':  # Q只取合并QKV输出行前1/3。
            end = weight.shape[0] // 3  # weight是[3D,D]，此处end=D。
            weight, bias = weight[:end], None if bias is None else bias[:end]  # 同步切权重和偏置。
        elif mode == 'kv':  # KV取输出行后2/3。
            start = weight.shape[0] // 3  # 从D行开始。
            weight, bias = weight[start:], None if bias is None else bias[start:]  # 输出变为2D。
        return F.linear(inputs, weight, bias), gate_decision  # 单线性输出及同一gate对象。


namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'Tensor': Tensor, 'Optional': Optional, 'Union': Union, 'Callable': Callable, 'partial': partial, 'TaskMoE': TraceTaskMoE, 'comm': NS(is_main_process=lambda: True)}  # 原类需要的全局名字。
exec(compile(ast.Module(body=classes, type_ignores=[]), '<original-moe-layer>', 'exec'), namespace)  # 运行原类定义，未改其方法体。
Attention = namespace['MoEAttentionBlock']  # 原注意力类对象。
Encoder = namespace['MoETransformerEncoderLayer']  # 原编码层类对象。


def config(enabled=True):  # 仅提供本检查用到的配置字段，关闭提示/融合/强制FP16。
    return NS(MOE=NS(MOE=enabled, MOE_TYPE='attribute', NUM_EXPERTS=2, TOP_K=2, CAPACITY_FACTOR=1, EVAL_MIN_CAPACITY=1, MIN_CAPACITY=1, NOISY_GATE_POLICY=None, USE_RTS=False, USE_TUTEL=False, FFN_SHARE_GATE_DECISION=True), MODEL=NS(BERT=NS(QKV_BIAS=False, UNIFY_QKV=True, SCALE_MULTI_BEFORE=False), PROMPT_EMBED=NS(DEEP_PROMPT=False), LAYER_SCALE_FP32=False), SOLVER=NS(FUSED_LAYERNORM=False, FORCE_LN_FP16=False, FORCE_SOFTMAX_FP16=False))  # NS支持原cfg属性访问。


cfg = config()  # 全局MoE开，以便原构造变量定义；先用稠密注意力分支检查实际数学。
attention = Attention(4, 2, 0.0, cfg, attn_moe=False).eval()  # D=4,h=2,d_h=2，Dropout=0，CPU原注意力。
with torch.no_grad():  # 参数赋值不建立梯度图，避免原地改叶子参数错误。
    attention.qkv_proj.weight.copy_(torch.eye(4).repeat(3, 1))  # Q/K/V都设恒等映射，便于独立手算。
    attention.dense.weight.copy_(torch.eye(4))  # 输出投影也恒等。
    attention.dense.bias.zero_()  # 消除输出偏置。
inputs = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 1.], [0., 1., 1., 0.]]])  # [B=1,L=4,D=4]确定输入。
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)  # 上三角True禁止未来key，保留对角和历史。
output, probabilities = attention(inputs, mask, None)  # 原类真实稠密注意力计算。
heads = inputs.reshape(1, 4, 2, 2).transpose(1, 2)  # 手算路径[B,h,L,d_h]，与原投影等价但独立组织公式。
manual_scores = heads @ heads.transpose(-1, -2) / math.sqrt(2)  # QKᵀ/sqrt(d_h)。
manual_probabilities = (manual_scores + mask[None, None] * -10000.0).softmax(-1)  # 按原有限负数mask协议手算softmax。
manual = (manual_probabilities @ heads).transpose(1, 2).reshape(1, 4, 4)  # 值聚合与头合并。
assert torch.allclose(output, manual)  # 实际原类应与手算一致。
assert torch.allclose(probabilities, manual_probabilities)  # 权重也逐项一致。
history_output, _ = attention(inputs[:, -2:], mask[-2:], None, history_states=inputs)  # query只取最后2项，key/value源完整4项。
assert torch.allclose(history_output, output[:, -2:])  # 原历史隐藏投影路径与完整注意力对应query行相同。
_, all_masked_probabilities = attention(inputs, torch.ones(4, 4, dtype=torch.bool), None)  # 整行全部遮盖的边界。
assert torch.allclose(all_masked_probabilities.sum(-1), torch.ones(1, 2, 4))  # 有限-10000仍归一化为总和1，不代表无人可见时权重全零。
TraceTaskMoE.calls.clear()  # 清除旧接口日志，专门检查历史Q/KV/dense的门控复用。
moe_attention = Attention(4, 2, 0.0, cfg, moe_layer=partial(TraceTaskMoE), attn_moe=True).eval()  # 原注意力搭配门控接口替身。
moe_output, _ = moe_attention(inputs[:, -2:], mask[-2:], None, history_states=inputs, moe_embedding=torch.ones(1, 8))  # 实际原attention调用三个专家接口位置。
assert [call['mode'] for call in TraceTaskMoE.calls] == ['q', 'kv', None]  # Q投影、KV投影、输出投影顺序。
assert len({call['gate_id'] for call in TraceTaskMoE.calls}) == 1  # 三处收到同一门控决策对象。
boundaries = {}  # 收集原Encoder边界异常类型，避免只凭阅读宣称运行表现。
try:  # 全局MoE关的构造路径。
    Encoder(4, 2, dropout=0, drop_path_ratio=0, cfg=config(False))  # 原代码仍引用未定义MoE_layer。
except UnboundLocalError as error:  # 精确捕获预期异常，不吞掉无关错误。
    boundaries['global_moe_off'] = type(error).__name__  # 保存已运行证据。
else:  # 若原行为改变，检查必须显式失败。
    raise AssertionError('expected undefined MoE_layer boundary')  # 不用旧结论掩盖新事实。
post = Encoder(4, 2, dropout=0, drop_path_ratio=0, cfg=cfg, norm_first=False)  # 原Post-LN，稠密FFN。
try:  # 检查属性kwargs传到norm2的原行为。
    post(inputs, moe_embedding=torch.ones(1, 8))  # norm2=nn.LayerNorm不接收moe_embedding。
except TypeError as error:  # 预期额外关键字导致TypeError。
    boundaries['post_ln_extra_kwargs'] = type(error).__name__  # 保存异常类型。
else:  # 未异常则原结构变化，应复核文档。
    raise AssertionError('expected post-LN kwargs boundary')  # 禁止假通过。
record = {'scope': '原注意力稠密局部运行；门控复用使用TraceTaskMoE接口替身；不是完整专家/模型运行', 'torch_version': torch.__version__, 'output_shape': list(output.shape), 'probability_shape': list(probabilities.shape), 'manual_max_error': (output-manual).abs().max().item(), 'history_max_error': (history_output-output[:, -2:]).abs().max().item(), 'all_masked_row_sum': all_masked_probabilities.sum(-1).tolist(), 'gate_reused_q_kv_dense': True, 'config_boundaries': boundaries}  # 结果与证据边界。
output_file = root / '复现/outputs/MoE注意力层检查.json'  # 输出路径，不混入论文指标。
output_file.parent.mkdir(parents=True, exist_ok=True)  # 创建目录。
output_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')  # 保存完整结果。
print(json.dumps(record, ensure_ascii=False, indent=2))  # 输出简短证据，未用真实训练数据。
