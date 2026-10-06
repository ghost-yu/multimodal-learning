# 【层内详解】torch 是 PyTorch 主库，提供 Tensor、张量矩阵运算和自动求导。本文件用 cat/zeros/matmul/logical_or 等；训练参数可通过 autograd 得梯度。
import torch
# 【层内详解】nn=neural network（神经网络）模块；常用 Module、Linear、LayerNorm、Dropout、Parameter、Identity，本层用它们构建可训练模块及残差分支。
from torch import nn
# 【层内详解】F 是 functional（函数式接口）别名；F.linear(x,weight,bias)直接使用给定权重算线性映射，F.softmax沿指定维转概率，F.relu为激活；不同于nn.Linear对象自行维护参数。
import torch.nn.functional as F

# 【层内详解】typing 是类型提示标准库；Optional允许None，Union允许多种类型，Callable表示可调用对象；Any为任意类型。本文件Any未实际使用，提示不自动检查运行输入。
from typing import Optional, Any, Union, Callable
# 【层内详解】导入Tensor类用于函数类型注解，表示PyTorch张量；并非创建了一个张量。
from torch import Tensor

# 【层内详解】点号表示相邻包的相对导入；get_activation将激活名字转为可调用函数；get_act_layer本文件未调用，不能据导入存在猜测它在本层实际生效。
from .create_act import get_act_layer, get_activation
# 【层内详解】timm 是视觉模型库；DropPath=随机深度，训练时按样本随机丢掉整个残差路径，与Dropout逐元素丢弃不同。这里是旧版导入路径，运行兼容需核对。
from timm.models.layers import DropPath
# 【层内详解】本仓库相邻layer_norm中的实现，用于配置选择的融合归一化；不是直接等同torch.nn.LayerNorm的实现细节。
from .layer_norm import LayerNorm
# 【层内详解】DeepPrompt=深层提示模块，在各层给key/value序列加入可学习提示token；不是用户输入的自然语言prompt字符串。
from .pe_encoder import DeepPrompt
# 【层内详解】TaskMoE=任务条件的混合专家模块，包裹线性层，返回(变换结果,门控决策)。实际属性来自kwargs的moe_embedding等；不是这里重新从图片分类。
from uniperceiver.task_moe.layer import TaskMoE
# 【层内详解】comm=communication通信工具，is_main_process判断主进程；此处还影响batch_first赋值，不能只把它当打印日志条件。
from uniperceiver.utils import comm
# 【层内详解】partial 是Python标准库的偏函数工具，提前绑定TaskMoE公共配置，后面只补hidden_size和expert；不立即执行训练。
from functools import partial
# 【层内详解】math 是数学标准库；sqrt求平方根，用1/sqrt(每头维度)缩放注意力分数，避免点积尺度随维度增大。
import math
# 【层内详解】FP16=16位浮点；导入项目的特定归一化实现，由FORCE_LN_FP16配置选择，不代表整层所有运算都一定FP16。
from uniperceiver.modeling.layers import FP16LayerNorm
# 【层内详解】autocast 是CUDA自动混合精度上下文；本处enabled=False关闭自动选择，然后显式.half()强制softmax输入为FP16。CPU/新版本适配尚需核查。
from torch.cuda.amp import autocast

# 【层内详解】MoE=Mixture of Experts（混合专家）；EncoderLayer是一个Transformer编码层，包含自注意力SA与前馈FFN两个残差分支。上方英文大段沿用PyTorch模板，不能视为本实现全部接口/快速路径保证。
class MoETransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:
        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)
        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    # 【层内详解】__constants__为TorchScript常量声明习惯；不意味着本自定义层已提供模板文档中的优化fast path或NestedTensor支持。
    __constants__ = ['batch_first', 'norm_first'] # we inherit this variable from pytorch's code for jit

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, drop_path_ratio: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_scale: bool = False, ls_init_values: float = 1e-3,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, cfg: dict = None, ffn_moe: bool = False, attn_moe: bool = False) -> None:

        # 【层内详解】注意只在主进程且batch_first=True时进入，把局部配置改False；其他进程可能保留True。但自定义注意力实际按[B,N,C]读取，不能根据模板batch_first注释交换轴。
        if batch_first and comm.is_main_process():
            print(f'set batch_first to \'False\' to support torch >= 1.12!')
            # 【层内详解】保留原代码的配置改写；它不把输入Tensor从[B,N,C]转成[N,B,C]。
            batch_first = False

        # 【层内详解】factory_kwargs=构造工厂关键字字典；当前是空，因此接口device/dtype并未装入此字典，没有据这两个参数自动设置所有层设备/精度。
        factory_kwargs = {}
        # 【层内详解】super调用父nn.Module初始化，使后续子模块、Parameter注册进入参数树；self是当前层实例。
        super(MoETransformerEncoderLayer, self).__init__()

        # 【层内详解】cfg=configuration配置对象，需支持cfg.MOE/cfg.MODEL/cfg.SOLVER属性；注解dict不代表普通Python dict直接支持点访问。
        self.cfg = cfg

        # The interface of nn.MultiheadAttention changed since torch 1.9.0.
        # 【层内详解】把版本字符串按点分开取主/次版本，如2.6.0→["2","6"]；此变量后续仅记录标志，实际自定义Attention没有用该标志选择torch.MultiheadAttention调用。
        _torch_version_main = torch.__version__.split('.')[:2]
        # 【层内详解】原比较要求主版本>=1且次版本>=9，会把2.6判旧接口，版本比较不等于按(major,minor)元组排序；这里只记录原边界。
        if (int(_torch_version_main[0]) >= 1) and (int(_torch_version_main[1])) >= 9:
            self._torch_nn_new_interface = True
        else:
            self._torch_nn_new_interface = False

        # for moe
        # 【层内详解】ffn=feed-forward network前馈网络，局部开关与全局MoE都开启才用专家线性层。
        self.ffn_moe = ffn_moe and self.cfg.MOE.MOE
        # 【层内详解】attn=attention注意力；保存有效开关。但创建self_attn时传的是原始attn_moe参数，需保持配置一致，不能假定自动全局关闭。
        self.attn_moe = attn_moe and self.cfg.MOE.MOE
        # 【层内详解】只有全局MoE开启才定义下方局部MoE_layer；关闭时之后仍引用它，可能UnboundLocalError，这个类并非所有非MoE设置都可独立初始化。
        if self.cfg.MOE.MOE:
            # assert self.ffn_moe and self.attn_moe
            # data-independent moe
            # 【层内详解】当前实现只接受attribute（属性）门控类型，论文其他门控对照不能据此认为全部在本类支持。
            if self.cfg.MOE.MOE_TYPE in ['attribute']:
                # 【层内详解】MoE_layer是绑定了专家数、top-k、噪声等配置的构造函数，后面为QKV/输出/FFN分别建TaskMoE实例。
                MoE_layer = partial(
                    TaskMoE,
                    # 【层内详解】E=专家总数；不同于每次选择的k。
                    num_experts=cfg.MOE.NUM_EXPERTS,
                    # 【层内详解】top-k决定每组条件选择几个专家，论文默认top-2；由TaskMoE/TopKGate决定归一化与融合。
                    k=cfg.MOE.TOP_K,
                    # 【层内详解】capacity=专家容量的通用MoE配置，具体属性融合分支是否用到须读TaskMoE，不能直接按传统逐token派发解释全部参数。
                    capacity_factor=cfg.MOE.CAPACITY_FACTOR,
                    # 【层内详解】保留原配置传递，字段名字含MIN而参数名factor，两者含义不能只凭名字认定相同，当前路线须结合底层实现。
                    eval_capacity_factor=cfg.MOE.EVAL_MIN_CAPACITY,
                    min_capacity=cfg.MOE.MIN_CAPACITY,
                    # 【层内详解】noisy gate是给门控加入噪声的策略，用于探索专家；评估时是否加噪声由底层training分支决定。
                    noisy_gate_policy=cfg.MOE.NOISY_GATE_POLICY,
                    # 【层内详解】RTS=Random Token Selection随机token选择，为通用MoE接口选项；不是本层新实现的图像采样。
                    use_rts=cfg.MOE.USE_RTS,
                    # 【层内详解】Tutel为MoE加速库选项，传入不代表本地已安装或当前属性融合路径必用它。
                    use_tutel=cfg.MOE.USE_TUTEL,
                    cfg=cfg,
                )
            else:
                raise NotImplementedError(f'{self.cfg.MOE.MOE_TYPE}')



        # 【层内详解】self_attn=self-attention自注意力模块，d_model为隐藏宽D，nhead为头数h；MoE作用在线性投影，不把softmax本身替换成专家。
        self.self_attn = MoEAttentionBlock(d_model, nhead, attention_probs_dropout_prob=dropout, cfg=cfg, moe_layer=MoE_layer, attn_moe=attn_moe)


        # 【层内详解】存配置标志；本自定义forward仍实际期待[B,N,D]，没有按此标志转置输入。
        self.batch_first = batch_first

        # Implementation of Feedforward model
        # 【层内详解】FFN第一层将每个token从D扩为F=dim_feedforward；nn.Linear作用在最后维，前面的B/N保持。
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        # 【层内详解】Dropout训练时逐元素随机置零并缩放，eval时关闭；此实例位于FFN激活后。
        self.dropout = nn.Dropout(dropout)
        # 【层内详解】FFN第二层F→D，回到残差可相加的隐藏宽度。
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # 【层内详解】有效FFN专家开关开启时，把两层线性分别包成TaskMoE；两层之间仍有非线性激活。
        if self.ffn_moe:
            # 【层内详解】expert是已有的D→F模板线性层，由TaskMoE复制成多专家；hidden_size给门控使用，不是直接把输出F改回D。
            self.linear1 = MoE_layer(hidden_size=d_model, expert=self.linear1)
            # 【层内详解】第二个F→D专家模块；虽然输入F，属性门控使用配置hidden_size，不能直接推断门控一定读取F维每token激活。
            self.linear2 = MoE_layer(hidden_size=d_model, expert=self.linear2)

        # 【层内详解】norm_first=True为Pre-LN，先归一化再算子层；False为Post-LN，在残差相加后归一化。
        self.norm_first = norm_first
        # 【层内详解】选择融合LayerNorm实现，减少归一化的算子开销；本文件不实现融合内核。
        if self.cfg.SOLVER.FUSED_LAYERNORM:
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        # 【层内详解】否则选择强制FP16归一化；常规分支使用nn.LayerNorm。归一化的精度会影响数值稳定性。
        elif self.cfg.SOLVER.FORCE_LN_FP16:
            self.norm1 = FP16LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = FP16LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        # 【层内详解】attention残差分支输出的逐元素dropout，与注意力概率dropout和FFN内部dropout是不同位置。
        self.dropout1 = nn.Dropout(dropout)
        # 【层内详解】FFN残差分支输出的逐元素dropout。
        self.dropout2 = nn.Dropout(dropout)

        # 【层内详解】随机深度作用在注意力整个残差输出；比例0时Identity原样返回。
        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # 【层内详解】FFN残差分支使用另一DropPath；不是把原输入x一起随机删除。
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # 【层内详解】LayerScale为每通道可学习残差缩放，可用小初值让层初始更接近恒等。
        self.layer_scale = layer_scale
        if self.layer_scale:
            # 【层内详解】gamma_1为注意力分支缩放[D]，Parameter注册训练参数；(d_model)没有逗号所以是整数，但torch.ones接受整数尺寸。
            self.gamma_1 = nn.Parameter(ls_init_values * torch.ones((d_model)),requires_grad=True)
            # 【层内详解】gamma_2为FFN分支缩放[D]，广播到[B,N,D]。
            self.gamma_2 = nn.Parameter(ls_init_values * torch.ones((d_model)),requires_grad=True)

        # Legacy string support for activation function.
        # 【层内详解】激活可传函数或名字字符串；isinstance是Python类型判断。
        if isinstance(activation, str):
            # 【层内详解】把relu/gelu等名字解析为函数；未识别名字的行为看create_act，不在这里自动纠错。
            activation = get_activation(activation)

        # 【层内详解】保存可调用激活，在FFN第一/二层之间使用。
        self.activation = activation

        # prompt embedding setup
        # 【层内详解】deep prompt在每个指定层插入学习的key/value提示序列；与任务属性门控编码不是同一个张量。
        self.deep_prompt = self.cfg.MODEL.PROMPT_EMBED.DEEP_PROMPT
        if self.deep_prompt:
            # 【层内详解】构造提示嵌入模块，forward动态取[B,P,D]提示；P为提示token数。
            self.deep_prompt_embedding = DeepPrompt(cfg)




    # 【层内详解】__setstate__在反序列化对象状态时使用，兼容老版本缺失activation字段，不是每步训练调用。
    def __setstate__(self, state):
        if 'activation' not in state:
            # 【层内详解】老状态未保存激活时补ReLU默认；不能据此把所有训练配置的激活都当ReLU。
            state['activation'] = F.relu
        super(MoETransformerEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                history_states: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        # 【层内详解】src=source输入序列[B,N,D]，x为后续变换变量；残差加法保留原层信息。
        x = src

        # 【层内详解】Pre-LN分支：历史输入也用norm1归一化，才能与当前已归一化特征处于同一注意力输入空间。
        if self.norm_first:
            # 【层内详解】history_states是历史隐藏状态，不是已投影的key/value缓存；None不做运算，其余按最后维归一化。
            history_states_norm = history_states if (history_states is None) else self.norm1(history_states)
            # 【层内详解】Pre-LN注意力残差：x+SA(LN(x))，kwargs继续带任务属性/sample_info等；输出仍[B,N,D]。
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, history_states=history_states_norm, **kwargs)
            # 【层内详解】Pre-LN前馈残差：x+FFN(LN(x))；新的x来自注意力残差。
            x = x + self._ff_block(self.norm2(x), **kwargs)
        else:
            # 【层内详解】Post-LN注意力先残差后LN；不在这里对历史做norm1。
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, history_states=history_states, **kwargs))
            # 【层内详解】保留原行为：kwargs被传给norm2而不是_ff_block。MoE所需属性因此可能丢失，普通LayerNorm收到额外键也会TypeError；不能默认Post-LN属性路径可用。
            x = self.norm2(x + self._ff_block(x), **kwargs)


        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], history_states: Optional[Tensor],
                  **kwargs) -> Tensor:

        # 【层内详解】有历史时按序列维拼历史与当前，形成注意力key/value源；之后仍重新做投影，不等同标准已投影KV缓存。
        if history_states is not None:
            kv = torch.cat(
                [history_states, x],
                # 【层内详解】拼接维1是token轴，要求输入[B,N,D]；B/D不变，key序列长度增加。
                dim=1
            )
            # TODO: changes for attn_mask and key_padding_mask
        else:
            # 【层内详解】无额外历史时以当前x同时产生Q/K/V。kv名字借用key/value，但这里保存的是投影前隐藏状态。
            kv = None

        if self.deep_prompt:

            # 【层内详解】从提示模块取[B,P,D]；函数显式batch_first=True再次说明当前计算按batch第一维组织。
            deep_prompt_embedding = self.deep_prompt_embedding(x, batch_first=True, **kwargs)
            # 【层内详解】Pre-LN分支：历史输入也用norm1归一化，才能与当前已归一化特征处于同一注意力输入空间。
            if self.norm_first:
                # 【层内详解】Pre-LN时提示与当前/历史使用同一norm1，保持输入空间一致。
                deep_prompt_embedding = self.norm1(deep_prompt_embedding)
            # 【层内详解】提示放在key/value序列最前，query仍为原x，输出token数不增加P。
            kv = torch.cat([deep_prompt_embedding, x], dim=1) if kv is None else torch.cat([deep_prompt_embedding, kv], dim=1)
            if 'sample_info' in kwargs:
                pe_length = deep_prompt_embedding.shape[1]
                # 【层内详解】pe=prompt embedding，此处在嵌套sample_info字典原地记录提示长度；kwargs外层复制并不隔离嵌套对象，这会影响调用者共享元数据。
                kwargs['sample_info']['pe_length'] = pe_length
            if attn_mask is not None:
                # 【层内详解】L=query长度，S=key长度；注意力mask必须能覆盖历史/提示扩展后的key轴。
                L, S = attn_mask.shape
                pe_length = deep_prompt_embedding.shape[1]  # length, bs, hidden_size
                # 【层内详解】给新增提示key补P列0，表示允许所有当前query读提示；原mask不变，列数增加。
                attn_mask = torch.cat([torch.zeros((L, pe_length), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask], dim=1)
            if key_padding_mask is not None:

                # 【层内详解】bs=batch size，pe_length=提示长度，取[B,P,D]前两轴；原邻近英文length,bs注释不符合实际batch_first排列。
                bs, pe_length = deep_prompt_embedding.shape[:2]

                key_padding_mask = torch.cat(
                    # 【层内详解】给提示补非padding标志0，与原key_padding_mask按列拼接。
                    [torch.zeros((bs, pe_length), dtype=key_padding_mask.dtype, device=key_padding_mask.device), key_padding_mask], dim=1)


        # 【层内详解】取注意力上下文结果，_表示丢弃注意力概率；kwargs继续传属性门控条件。
        x, _ = self.self_attn(x, history_states=kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask, **kwargs)
        # 【层内详解】对注意力输出做dropout和随机深度；残差加法在外层forward完成。
        x = self.drop_path1(self.dropout1(x))
        if self.layer_scale:
            if self.cfg.MODEL.LAYER_SCALE_FP32:
                # 【层内详解】用FP32参数gamma_1乘残差，可能提升运算输出精度；具体类型由PyTorch类型提升规则决定。
                x = self.gamma_1 * x
            else:
                # 【层内详解】先把gamma转换为x精度再乘，避免该处因FP32参数改变分支dtype；to转换仍可传梯度。
                x = self.gamma_1.to(x.dtype) * x
        return x


    # feed forward block
    def _ff_block(self, x: Tensor, **kwargs) -> Tensor:
        # 【层内详解】有效FFN专家开关开启时，把两层线性分别包成TaskMoE；两层之间仍有非线性激活。
        if self.ffn_moe:
            # 【层内详解】FFN第一专家线性返回隐藏结果[B,N,F]与路由决策（专家索引/权重）。gate_decision不是分类标签。
            x, gate_decision = self.linear1(x, **kwargs)
            # 【层内详解】配置决定FFN第二线性是否复用第一线性门控；不共享时第二模块自己重算。
            if not self.cfg.MOE.FFN_SHARE_GATE_DECISION:
                # 【层内详解】None触发后续模块重新门控；不是把专家结果置零。
                gate_decision = None
            # 【层内详解】先激活/Dropout再第二专家映射F→D，可复用第一路由；不能把非线性的整个FFN合并成一个线性权重。
            x, _ = self.linear2(self.dropout(self.activation(x)), gate_decision=gate_decision, **kwargs)
        else:
            # 【层内详解】未启用FFN专家的普通两层MLP：Linear→激活→Dropout→Linear。
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # 【层内详解】FFN输出再做残差分支dropout与随机深度。
        x = self.drop_path2(self.dropout2(x))
        if self.layer_scale:
            if self.cfg.MODEL.LAYER_SCALE_FP32:
                # 【层内详解】FFN分支LayerScale，每通道gamma_2，与注意力gamma_1不同参数。
                x = self.gamma_2 * x
            else:
                # 【层内详解】匹配FFN结果dtype后再缩放，保留梯度链。
                x = self.gamma_2.to(x.dtype) * x
        return x


# 【层内详解】这是实际的多头注意力实现，输入[B,L,D]，包含QKV投影、softmax、值聚合和输出投影；专家化发生在QKV/dense两个线性位置。
class MoEAttentionBlock(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, cfg, moe_layer=None, attn_moe=False):
        super(MoEAttentionBlock, self).__init__()
        # 【层内详解】cfg=configuration配置对象，需支持cfg.MOE/cfg.MODEL/cfg.SOLVER属性；注解dict不代表普通Python dict直接支持点访问。
        self.cfg = cfg
        # 【层内详解】隐藏宽D必须可整除头数h；%取余，若非0报错，避免每头宽不为整数。
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention " "heads (%d)" % (hidden_size, num_attention_heads))

        # 【层内详解】hidden_size=D，全隐藏宽；该字段还用于切QKV权重的输出行。
        self.hidden_size = hidden_size

        # 【层内详解】num_attention_heads=h，独立注意力头数；不等于专家数E。
        self.num_attention_heads = num_attention_heads
        # 【层内详解】每头宽d_h=D/h。int转整数，前面已经验证可整除。
        self.attention_head_size = int(hidden_size / num_attention_heads)
        # 【层内详解】all_head_size=h*d_h=D，合并所有头后的宽度。
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 【层内详解】Q=Query询问，K=Key索引，V=Value内容；配置是否给QKV线性层加偏置。
        self.qkv_bias = cfg.MODEL.BERT.QKV_BIAS

        # 【层内详解】unify_qkv=True将Q/K/V三个映射放在一个输出3D的线性层，便于一起专家融合；不代表Q/K/V权重相同。
        self.unify_qkv = cfg.MODEL.BERT.UNIFY_QKV

        if not cfg.MODEL.BERT.UNIFY_QKV:
            self.query = nn.Linear(hidden_size, self.all_head_size, bias=self.qkv_bias)
            self.key = nn.Linear(hidden_size, self.all_head_size, bias=self.qkv_bias)
            self.value = nn.Linear(hidden_size, self.all_head_size, bias=self.qkv_bias)
        else:
            # 【层内详解】proj=projection投影，D→3D，最后维依次存Q/K/V三段。权重形状[3D,D]。
            self.qkv_proj = nn.Linear(hidden_size, self.all_head_size * 3, bias=self.qkv_bias)

        # 【层内详解】dense=全连接输出投影，合并头后的D→D，不是softmax的分类输出。
        self.dense = nn.Linear(hidden_size, self.all_head_size)

        # 【层内详解】在注意力概率上做Dropout，训练时各连接权重可能丢弃并缩放，行和不一定仍为1；eval时不变。
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.attn_moe = attn_moe
        if self.attn_moe:
            if not cfg.MODEL.BERT.UNIFY_QKV:
                raise NotADirectoryError('use UNIFY_QKV=True please')
            else:
                # 【层内详解】包成QKV专家层，每个专家都包含D→3D线性权重；输入组条件相同可先融合参数再一次投影。
                self.qkv_proj = moe_layer(hidden_size=hidden_size, expert=self.qkv_proj)
                # 【层内详解】输出投影也专家化，forward复用QKV的gate_decision，不重新按注意力结果每token路由。
                self.dense = moe_layer(hidden_size=hidden_size, expert=self.dense)

        # 【层内详解】决定先缩放Q再矩阵乘，还是点积后除sqrt(d_h)；数学等价，但浮点舍入和溢出风险可不同。
        self.scale_multi_before = cfg.MODEL.BERT.SCALE_MULTI_BEFORE

    # 【层内详解】将Q/K/V从[B,L,D]改为[B,h,L,d_h]，便于每个头独立算注意力；命名transpose表示轴交换。
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        # 【层内详解】*展开新尺寸元组，先把最后D拆成[h,d_h]，得到[B,L,h,d_h]；view要求内存步幅允许。
        x = x.view(*new_x_shape)

        # 【层内详解】创建轴顺序[0,1,2,3]，用下行交换token轴与head轴。
        shape_list = list(range(len(new_x_shape)))
        # 【层内详解】多目标赋值同时交换倒数2/3轴索引，避免先覆盖一项；四维时得到[0,2,1,3]。
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        # 【层内详解】permute按指定轴顺序返回视图[B,h,L,d_h]，没有更改数据值。
        return x.permute(shape_list)
        #return x.permute(0, 2, 1, 3)

    # 【层内详解】hidden_states是当前query来源[B,L,D]；history_states若给出，是完整key/value隐藏源[B,S,D]；两种mask分别为[L,S]和[B,S]。
    def forward(self, hidden_states, attn_mask, key_padding_mask, history_states=None,  **kwargs):
        if attn_mask is None and key_padding_mask is None:
            attention_mask = None
        else:
            # attn_mask [L, S]  key_padding_mask[N, S]
            if attn_mask is not None and key_padding_mask is not None:
                # 【层内详解】把[L,S]→[1,L,S]与[B,S]→[B,1,S]广播逻辑或，得[B,L,S]；True=禁止，任一mask禁止就遮住。
                attention_mask = torch.logical_or(attn_mask.unsqueeze(0).bool(), key_padding_mask.unsqueeze(1).bool())
            elif attn_mask is not None:
                # 【层内详解】只有序列mask时添加batch共享轴[1,L,S]。当前接口按禁止0/1解释，不是任意加性-inf mask。
                attention_mask = attn_mask.unsqueeze(0)
            else:
                # 【层内详解】只有padding mask时[B,S]→[B,1,S]，对所有query共享无效key位置。
                attention_mask = key_padding_mask.unsqueeze(1)
        if attention_mask is not None:
            # 【层内详解】再加head共享轴，禁止位置加很大的负数-10000；有限负数不保证全行遮盖时概率全0，softmax仍可能归一化为非零。
            attention_mask = attention_mask.unsqueeze(1) * -10000.0

        if self.unify_qkv:
            if history_states is None:

                # 【层内详解】局部B=batch，N=当前token数，C=隐藏宽D；变量只在无历史分支解包，避免按模板[N,B,C]传错。
                B, N, C = hidden_states.shape
                if self.attn_moe:
                    # qkv, _, _ = self.self.qkv_proj(hidden_states)
                    # 【层内详解】此处变量hidden_states被覆盖为QKV投影[B,N,3D]；它已经不是投影前D维状态，路由决策随后复用。
                    hidden_states, gate_decision = self.qkv_proj(hidden_states, **kwargs)
                    # 【层内详解】chunk沿最后3D分成Q/K/V三份，每份[B,N,D]；mixed表示头还未分开。
                    mixed_query_layer, mixed_key_layer, mixed_value_layer =hidden_states.chunk(3, dim=-1)
                else:
                    # 【层内详解】非专家路径也用合并QKV层后切三份；权重不同，只共享一次调用。
                    mixed_query_layer, mixed_key_layer, mixed_value_layer = self.qkv_proj(hidden_states).chunk(3, dim=-1)

            else:
                # usually inference with history embedding
                if self.attn_moe:

                    # 【层内详解】有历史时当前输入只取Q专家权重段D行，返回[B,L,D]与门控。mode是底层专家参数切片选择。
                    mixed_query_layer, gate_decision = self.qkv_proj(hidden_states, mode='q', **kwargs)

                    # 【层内详解】完整历史源只取K/V两段2D，复用当前Q路由；[0]取结果张量，随后切为K/V。它重新投影隐藏历史，不是直接复用投影缓存。
                    history_states = self.qkv_proj(history_states, mode='kv', gate_decision=gate_decision, **kwargs)[0]
                    # 【层内详解】历史投影[B,S,2D]切成两个[B,S,D]，S可大于当前query长度L。
                    mixed_key_layer, mixed_value_layer = history_states.chunk(2, dim=-1)

                else:
                    # query
                    # 【层内详解】合并QKV权重的Q输出行从0开始；下划线表示临时局部变量。
                    _start = 0
                    # 【层内详解】Q输出行到D（不含D），即weight[0:D,:]。
                    _end = self.hidden_size
                    mixed_query_layer = F.linear(hidden_states,
                                                 self.qkv_proj.weight[_start:_end, :],
                                                 bias=None if self.qkv_proj.bias is None else self.qkv_proj.bias[_start:_end])

                    # key and value
                    # torch.equal(key, value)
                    # 【层内详解】K/V输出行从D开始，直到3D；同变量被重赋值，当前已不是Q的起点0。
                    _start = self.hidden_size
                    mixed_key_layer, mixed_value_layer = F.linear(history_states,
                                                                  self.qkv_proj.weight[_start:, :],
                                                                  bias=None if self.qkv_proj.bias is None else self.qkv_proj.bias[_start:]).chunk(
                                                                      2, dim=-1)


        else:
            raise NotImplementedError('please use unify qkv_proj')

        # 【层内详解】Q拆头[B,h,L,d_h]，当前query长度为L。
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # 【层内详解】K拆头[B,h,S,d_h]，完整key长度为S。
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # 【层内详解】V拆头[B,h,S,d_h]；这里Value V与词表大小V无关，本层没有词表输出。
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.scale_multi_before:
            # 【层内详解】Q先除sqrt(d_h)，K转置为[B,h,d_h,S]，批量matmul得[B,h,L,S]。
            attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        else:
            # 【层内详解】另一分支先算QKᵀ，后面再缩放；matmul对batch/head广播。
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # 【层内详解】按每头维度缩放，不是按头数h或序列长度缩放。
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            # 【层内详解】mask按batch/head/query轴广播加到分数，遮住不允许读的key。
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # 【层内详解】强制softmax输入为FP16；与常见先转FP32增强稳定性的做法不同，不要按名称猜反。
        if self.cfg.SOLVER.FORCE_SOFTMAX_FP16:
            # 【层内详解】上下文关闭自动混合精度，不关闭梯度；下一行.half显式选精度。
            with autocast(enabled=False):
                # 【层内详解】softmax沿最后key轴S；结果FP16可能与value_layer不同dtype，配置要协调，不能把此设置当所有环境可用。
                attention_probs = F.softmax(attention_scores.half(), dim=-1)
        else:
            # 【层内详解】普通分支创建无参数Softmax模块并调用，沿key维输出权重[B,h,L,S]。
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 【层内详解】概率Dropout用于训练正则化；eval时行和约1，train时单次结果不一定。
        attention_probs = self.dropout(attention_probs)

        # 【层内详解】权重[B,h,L,S]乘Value[B,h,S,d_h]→上下文[B,h,L,d_h]，每个query汇总所有可见key的内容。
        context_layer = torch.matmul(attention_probs, value_layer)

        # 【层内详解】再次组织轴顺序，这次要把head和token交换回[B,L,h,d_h]。
        shape_list = list(range(len(context_layer.shape)))
        # 【层内详解】多目标赋值同时交换倒数2/3轴索引，避免先覆盖一项；四维时得到[0,2,1,3]。
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        # 【层内详解】换轴后contiguous使存储连续，下一步才方便view合并头。
        context_layer = context_layer.permute(shape_list).contiguous()
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 【层内详解】前面[B,L]保留，最后[h,d_h]合并为D；单元素元组需要尾逗号。
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        # 【层内详解】合并注意力头，结果[B,L,D]，输出token数等于query数L。
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.attn_moe:
            # 【层内详解】专家输出投影复用QKV路由，返回最终[B,L,D]；_丢弃再次返回的门控。
            context_layer, _ = self.dense(context_layer, gate_decision=gate_decision, **kwargs)
        else:
            # 【层内详解】非专家输出投影D→D；它不改变序列长度。
            context_layer = self.dense(context_layer)

        # 【层内详解】返回最终上下文和注意力权重；外层_sa_block只用前者，权重可用于检查形状但不直接等同可靠解释。
        return context_layer, attention_probs
