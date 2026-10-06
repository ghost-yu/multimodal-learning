# 教学副本：保持官方仓库现有可执行语句不变，只增补给初学者的中文解释。
# 结构对照：原始文件 BLIP/models/med.py；建议同时阅读 项目讲解/BLIP-从零重讲 系列。
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
'''

# Python 标准数学库；注意力分数按 sqrt(每头维数) 缩放时使用。
import math
# Python 路径/系统库；本文件继承了 Transformers BERT 的兼容性代码，主前向路径不靠它。
import os
# Python 警告工具；不是神经网络层或损失函数。
import warnings
# dataclass 是定义数据容器的装饰器；当前 BLIP 主前向路径不调用它。
from dataclasses import dataclass
# 类型提示：Optional[X] 允许 None；Tuple 表示元组，不会自动检查张量形状。
from typing import Optional, Tuple

# PyTorch 张量与自动求导的主命名空间，如 torch.matmul、torch.cat、torch.arange。
import torch
# 导入类型/命名空间：Tensor 是张量类型，device 表设备，dtype 表数值类型，nn 提供网络层。
from torch import Tensor, device, dtype, nn
# 梯度检查点工具；前向少保存激活，反向重新计算，用计算时间换显存。
import torch.utils.checkpoint
# 重复导入同一 nn 名称，不会复制一套网络参数；原代码保留的写法。
from torch import nn
# 交叉熵损失类；LM 让每个位置的词表 logits 与真实下一词 ID 比较。
from torch.nn import CrossEntropyLoss
# F=functional，函数式网络运算；本文件用 softmax、dropout 等无需单独建层的函数。
import torch.nn.functional as F

# ACT2FN 是激活名字到函数的字典；如配置 hidden_act=gelu 时选 GELU 函数。
from transformers.activations import ACT2FN
# 下方 ModelOutput 是 Hugging Face 输出容器基类，允许同时按属性和元组位置读取结果。
from transformers.file_utils import (
    ModelOutput,
)
# 下方多种 Output 类是不同任务的返回结构；主路径常取 last_hidden_state 或 loss。
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 下方导入预训练模型基类、分块前馈和剪注意力头工具；只有基类/分块在主路径常见。
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
# 使用 Transformers 自带日志器，和 Python 标准 logging 类似，用于提醒兼容情况。
from transformers.utils import logging
# BertConfig 保存层数、隐藏维、注意力头数、词表大小等设置；可从 JSON 加载。
from transformers.models.bert.configuration_bert import BertConfig


# __name__ 是本 Python 模块名；logger 是当前文件专用日志器，不参与前向计算。
logger = logging.get_logger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""
    # 作用：把整数 token ID 变成 [B,T,D] 的连续向量，再加位置向量。

    def __init__(self, config):
        super().__init__()
        # 词表中的每个 token ID 对应一个可训练的 D 维向量。
        # Embedding 是可训练查表层；输入 token 整数 ID [B,T]，输出词向量 [B,T,Dt]。
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 第 0、1、2... 个文字位置也有各自可训练的位置向量。
        # 为文字位置 0、1、2... 分别存可训练向量，让模型感知词序。
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 归一化每个 token 的特征，eps 防止除以接近零的方差。
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 在训练时随机置零部分值；概率来自配置，eval() 模式下停止丢弃。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # position_ids 是固定整数序列，随模型搬设备，但不会被优化器更新。
        # position_ids 是固定位置序号，随模型搬设备但不由优化器更新；arange 生成 0..最大长度-1。
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # getattr(obj,name,default) 读属性；配置缺少该项时默认绝对位置向量。
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # 保存构造时的 BertConfig；forward 以后会从 self.config 读取维度、开关和默认值。
        self.config = config

    def forward(
        self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 一般用 input_ids；若上游已经算好向量，也可直接用 inputs_embeds。
        if input_ids is not None:
            # BertModel 的 input_ids 为 [B,T] 整数词 ID；从它推断 batch 与序列长度。
            input_shape = input_ids.size()
        else:
            # 若已直接传词向量 [B,T,Dt]，去掉最后的特征宽度，得到 [B,T]。
            input_shape = inputs_embeds.size()[:-1]

        # 第 1 维是 token 序列长度 T。
        # input_shape 的第 1 维是本批文本 token 长度 T，不是 batch 大小 B。
        seq_length = input_shape[1]

        # 调用者未显式提供位置序号时，从已注册的连续 position_ids buffer 切取。
        if position_ids is None:
            # 生成时跳过历史缓存已用过的位置号，保证新 token 接续正确位置。
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 未提供现成向量时，查词向量表。
        if inputs_embeds is None:
            # 若上游给的是整数 ID，就在词向量表中查出 [B,T,Dt]；inputs_embeds 此后指向向量。
            inputs_embeds = self.word_embeddings(input_ids)

        # 这里 embeddings 指当前文字向量 [B,T,Dt]，不再指整数 ID。
        embeddings = inputs_embeds

        # 绝对位置编码与词向量逐元素相加，形状不变。
        if self.position_embedding_type == "absolute":
            # nn.Embedding 用整数位置号查可训练位置向量；输出与词向量同宽。
            position_embeddings = self.position_embeddings(position_ids)
            # 每个词向量加对应位置向量；+= 更新局部 embeddings，不是增加 token 数。
            embeddings += position_embeddings
        # LayerNorm 沿每个 token 的特征维归一化，稳定后续层的数值范围。
        embeddings = self.LayerNorm(embeddings)
        # 训练期随机置零部分嵌入分量，形状仍为 [B,T,Dt]。
        embeddings = self.dropout(embeddings)
        # 输出 [B,T,Dt] 的文字嵌入，已合并词/位置向量并做归一化和 Dropout。
        return embeddings


class BertSelfAttention(nn.Module):
    """文字内部注意力，或以文字为 Q、图像为 K/V 的交叉注意力。

    Q = Query（查询），K = Key（键），V = Value（值）。自注意力时三者
    都来自文字；交叉注意力时 Q 来自文字、K/V 来自视觉编码器的 patch。
    """
    def __init__(self, config, is_cross_attention):
        super().__init__()
        # 保存构造时的 BertConfig；forward 以后会从 self.config 读取维度、开关和默认值。
        self.config = config
        # 普通多头注意力要求隐藏宽度能被头数整除，以便每头获得相同维度。
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        # 注意力头数 H；例如 Dt=768、H=12 时每头 d=64。
        self.num_attention_heads = config.num_attention_heads
        # D/H 是每个注意力头的维数；例如 768/12=64。
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 重新合并 H 个头后的总宽度，通常等于文字隐藏宽度 Dt。
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # nn.Linear(D,D) 是可训练线性层；Query 总是从当前文字 token 产生。
        # Query 始终来自当前文字 hidden_states；Query 是每个文字 token 要寻找的信息。
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # 构造器开关：True 时 Key/Value 的输入来自图像，宽度是 encoder_width。
        if is_cross_attention:
            # 跨模态时，Key/Value 来自 ViT 输出，输入维度是 encoder_width。
            # 交叉注意力时 Key 的输入宽度是视觉 token Dv，来自 encoder_hidden_states=image_embeds。
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            # 交叉注意力时 Value 同样来自图像 token，汇总后回到文字位置。
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            # 纯文字自注意力时，Q/K/V 都从文字 token 产生。
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            # 普通文字自注意力的 Value 由文字向量产生，故输入宽度为 hidden_size。
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 这里 Dropout 作用在注意力概率上，不是输入词向量；概率来自 JSON 配置。
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # getattr(obj,name,default) 读属性；配置缺少该项时默认绝对位置向量。
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 只有配置选择相对位置时才创建距离向量表；默认 absolute 路径不会进入。
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 保存支持的最长文字位置，随后用来把负距离偏移为非负 Embedding 索引。
            self.max_position_embeddings = config.max_position_embeddings
            # 相对位置分支把两位置差值映成每头维度向量，表大小覆盖正负最大距离。
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        # 默认不保存注意力图；显式打开才额外保存 map/梯度，避免日常训练占内存。
        self.save_attention = False   
            
    def save_attn_gradients(self, attn_gradients):
        # 注意力可视化 hook 写入梯度；它是中间观察值，不是可训练 Parameter。
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        # 读取最近一次保存在对象上的注意力梯度。
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        # 把 softmax 注意力概率暂存为属性，供可视化代码查询。
        self.attention_map = attention_map
        
    def get_attention_map(self):
        # 返回已记录的注意力概率图；未记录时可能是 None。
        return self.attention_map
    
    def transpose_for_scores(self, x):
        # [B,T,D] -> [B,T,H,D/H] -> [B,H,T,D/H]，分出多头维度。
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # * 将形状元组拆成 view 的多个参数；把总特征宽度拆成 [头数,每头宽度]。
        x = x.view(*new_x_shape)
        # 把 [B,T,H,d] 的维度次序改成 [B,H,T,d]，便于批量 Q×K 转置矩阵乘法。
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # hidden_states 是文字 token；先把它映射为 Query。
        # hidden_states 是当前层文字向量 [B,T,Dt]；mixed 表示暂未拆分多个注意力头。
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # 上游传了图像特征，表示现在允许文字读取图像。
        # encoder_hidden_states 由上层传入图像 token；非 None 才执行“文字读图像”的交叉注意力。
        is_cross_attention = encoder_hidden_states is not None

        # 构造器开关：True 时 Key/Value 的输入来自图像，宽度是 encoder_width。
        if is_cross_attention:
            # K/V 从图像 token 而非文字 token 生成。
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            # 交叉注意力的 V 从图像 token 产生，拆多头后为 [B,H,P,d]。
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            # 图像有效位掩码替代文字掩码，控制可被读取的 K/V 位置。
            attention_mask = encoder_attention_mask
        # 逐词生成缓存分支：以前文字的 K/V 已经算过，本次只追加新 token 的 K/V。
        elif past_key_value is not None:
            # 生成文字时复用已算过的历史 K/V，避免每次重算前文。
            # 文字自注意力的 K 来自当前层文字；交叉注意力分支则改用图像。
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # 纯文字注意力或生成缓存时，V 来自当前文字 hidden_states。
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # 第 2 维是序列长度；历史 Key 接当前 Key，形成可读取的全部过去位置。
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            # Value 也沿序列维追加，必须与 Key 的历史长度和顺序一致。
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # 文字自注意力的 K 来自当前层文字；交叉注意力分支则改用图像。
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # 纯文字注意力或生成缓存时，V 来自当前文字 hidden_states。
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Query 也切成多头；形状 [B,H,Tq,D/H]。
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 把本层当前完整 K/V 打包返回，生成下一词时可复用。
        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # Q × K^T 得到 [B,H,Tq,Tk]，每个文字位置对全部可读位置打分。
        # Q×Kᵀ：自注意力得 [B,H,T,T]；交叉注意力得 [B,H,T,P]。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 只有配置选择相对位置时才创建距离向量表；默认 absolute 路径不会进入。
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 相对位置分支要知道文字序列 T；size()[1] 读第 1 维。
            seq_length = hidden_states.size()[1]
            # 左位置列向量 [T,1]；arange 产生整数 0..T-1，view 改形状供广播。
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            # 右位置行向量 [1,T]，与左列广播相减形成两两相对位置 [T,T]。
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            # 每个查询位置减被读取位置，正/负号区分相对方向。
            distance = position_ids_l - position_ids_r
            # Embedding 索引不能为负，所以给位置差整体加偏移，使其落到 0 开始的范围。
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            # 把位置向量转到 Q 的数值类型，避免半精度和全精度混算。
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            # 只给 Query 侧加入相对位置分数；另一配置 relative_key_query 会同时算两侧。
            if self.position_embedding_type == "relative_key":
                # einsum 指定维度求和：文字 Query 与相对位置向量点积，得额外 [B,H,T,T] 分数。
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                # 把内容相似度与相对位置分数逐元素相加，然后才做 softmax。
                attention_scores = attention_scores + relative_position_scores
            # 相对位置同时影响 Query 和 Key 两侧分数的兼容分支。
            elif self.position_embedding_type == "relative_key_query":
                # 按 einsum 维度约定把 Query 与相对位置向量相乘求和，得到 [B,H,T,T]。
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                # Key 侧也与相对位置向量点积；这里按原 BERT 兼容公式计算。
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                # 把内容 QK 分数与两种相对位置分数相加，然后才做 mask/softmax。
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # 缩放点积分数，防止 softmax 在高维时过度极端。
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 若上层传入 mask，就将不可读位置的巨大负数加到注意力原始分数上。
        if attention_mask is not None:
            # padding 或未来词的分数会加很大的负数，softmax 后近似为 0。
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # Softmax 沿最后一维归一：一个 Query 读取所有 Key 的权重和为 1。
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # 仅图文交叉注意力且打开记录开关时保存 map 与梯度；训练主路径默认不记录。
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)         

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            # head_mask 可关闭某些注意力头；与概率逐元素相乘后相应头贡献为 0。
            attention_probs_dropped = attention_probs_dropped * head_mask

        # 按注意力权重对 V 加权求和，得到各文字位置的新信息。
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        # 从 [B,H,T,d] 调为 [B,T,H,d]；contiguous 整理内存供随后 view 合并头维。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 保留 [B,T]，再把 [H,d] 合成总宽度 H×d=Dt，得到目标 [B,T,Dt]。
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 合并多头，恢复 [B,Tq,D]。
        context_layer = context_layer.view(*new_context_layer_shape)

        # 按开关决定是否额外返回注意力概率；(x,) 的逗号表示单元素元组。
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 将本层 K/V 缓存加到返回元组末尾，供解码器后续 token 使用。
        outputs = outputs + (past_key_value,)
        # outputs 通常是元组，第 0 项是本层主输出，后面可带注意力概率和 K/V 缓存。
        return outputs


class BertSelfOutput(nn.Module):
    """注意力后的线性投影、残差连接和 LayerNorm。"""
    def __init__(self, config):
        super().__init__()
        # dense 是全连接线性层，输入输出均为 Dt；在不同 Head 类中各有独立参数。
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 归一化每个 token 的特征，eps 防止除以接近零的方差。
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 在训练时随机置零部分值；概率来自配置，eval() 模式下停止丢弃。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states 是注意力新信息；input_tensor 是进入注意力前的旧信息。
        # 词预测前的内部变换：每个 token 的 Dt 维经过线性层。
        hidden_states = self.dense(hidden_states)
        # 训练时对当前层输出做 Dropout；hidden_states 仍是文字位置向量。
        hidden_states = self.dropout(hidden_states)
        # 残差相加可保留旧信息，并让深层网络更易训练。
        # 残差相加后归一化；两项形状同为 [B,T,Dt]。
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回本辅助层处理后的文字 token；张量形状与本层定义的输入/输出宽度对应。
        return hidden_states


class BertAttention(nn.Module):
    """把注意力计算和其后的投影/残差包装成一个模块。"""
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # is_cross_attention 决定 K/V 读取文字还是图像。
        self.self = BertSelfAttention(config, is_cross_attention)
        # BertAttention 的输出子层负责线性投影、Dropout 与残差 LayerNorm。
        self.output = BertSelfOutput(config)
        # 记录已经删除的注意力头编号，避免重复剪同一个头；主 BLIP 训练一般不剪。
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 剪枝请求为空时直接退出，不改注意力层参数。
        if len(heads) == 0:
            # 提前退出当前剪枝函数；没有要剪的头就无需改任何参数。
            return
        # 计算要删除的头及保留的线性层通道索引；用于模型剪枝的辅助路径。
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        # 用保留通道重建 Key 线性层；剪头后参数形状会改变。
        self.self.key = prune_linear_layer(self.self.key, index)
        # Value 也按相同头索引裁剪，与 Key 保持对齐。
        self.self.value = prune_linear_layer(self.self.value, index)
        # 输出投影的输入通道要同步减少，dim=1 指 Linear 权重的输入维。
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        # 剪头后重新计算总头宽度，供后续 reshape 使用。
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        # 集合 union 记录新剪掉的头编号，防止以后重复剪。
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 进入 BertSelfAttention；它根据是否传图像 token 选择自注意力或交叉注意力。
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # self_outputs[0] 是加权后的 token 信息，再与原 hidden_states 做残差。
        attention_output = self.output(self_outputs[0], hidden_states)
        # 把主输出放第 0 项，后面保留注意力图和缓存，保持上层解包协议。
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # outputs 通常是元组，第 0 项是本层主输出，后面可带注意力概率和 K/V 缓存。
        return outputs


class BertIntermediate(nn.Module):
    """FFN 的第一层：扩大维度并应用激活函数。"""
    def __init__(self, config):
        super().__init__()
        # 将 D 维扩展为 intermediate_size，BERT-Base 通常是 768 -> 3072。
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 若配置的激活是名称字符串，如 gelu，就从 ACT2FN 查函数。
        if isinstance(config.hidden_act, str):
            # 从字符串如 gelu 查出对应激活函数，后续像普通函数一样调用。
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 若上游直接传函数对象，则无需按字符串查字典，直接保存可调用对象。
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # 每个 token 独立做线性层和非线性激活。
        # 词预测前的内部变换：每个 token 的 Dt 维经过线性层。
        hidden_states = self.dense(hidden_states)
        # FFN 中间层应用非线性；没有它，两层线性相连仍等价于一层线性。
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回本辅助层处理后的文字 token；张量形状与本层定义的输入/输出宽度对应。
        return hidden_states


class BertOutput(nn.Module):
    """FFN 的第二层：压回隐藏维度，再做残差与归一化。"""
    def __init__(self, config):
        super().__init__()
        # 将中间维度压回 D，方便与输入残差相加。
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 归一化每个 token 的特征，eps 防止除以接近零的方差。
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 在训练时随机置零部分值；概率来自配置，eval() 模式下停止丢弃。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # input_tensor 是 FFN 前的旧 token 向量，提供残差路径。
        # 词预测前的内部变换：每个 token 的 Dt 维经过线性层。
        hidden_states = self.dense(hidden_states)
        # 训练时对当前层输出做 Dropout；hidden_states 仍是文字位置向量。
        hidden_states = self.dropout(hidden_states)
        # 残差相加后归一化；两项形状同为 [B,T,Dt]。
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回本辅助层处理后的文字 token；张量形状与本层定义的输入/输出宽度对应。
        return hidden_states


class BertLayer(nn.Module):
    """一层 MED：文字自注意力 → 可选图文交叉注意力 → FFN。"""
    def __init__(self, config, layer_num):
        super().__init__()
        # 保存构造时的 BertConfig；forward 以后会从 self.config 读取维度、开关和默认值。
        self.config = config
        # 前馈网络按 token 序列分块的大小；0 通常表示不分块。
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 张量 [B,T,Dt] 的第 1 维是序列长度，分块应沿这一维进行。
        self.seq_len_dim = 1
        # 首先始终让文字 token 互相读取。
        # 这套注意力总在文字 token 之间工作；即使多模态模式，也先做文字自注意力。
        self.attention = BertAttention(config)
        # layer_num 是这层在 BertEncoder 中的编号，来自创建 ModuleList 时的 i。
        self.layer_num = layer_num          
        # 配置允许才建立图文交叉注意力层；实际 forward 还要看 mode。
        if self.config.add_cross_attention:
            # 配置允许时准备一套独立交叉注意力参数，用于读取图像 token。
            # 构建独立的图文注意力参数；只有 forward 的 mode=multimodal 时才真正调用。
            self.crossattention = BertAttention(config, is_cross_attention=self.config.add_cross_attention)
        # FFN 的第一半：Linear 把 Dt 扩到 intermediate_size 并激活。
        self.intermediate = BertIntermediate(config)
        # FFN 的第二半：压回 Dt、Dropout、残差、LayerNorm。
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        mode=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 第一步：只在文字序列内部做 self-attention。
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 取当前层更新后的全部文字 token 表示。
        # 取本层文字自注意力的主要输出 [B,T,Dt]；下面可能继续读图像。
        attention_output = self_attention_outputs[0]

        # 除了第 0 项的主输出和最后一项 K/V 缓存，中间可能包含注意力图；先保留。
        outputs = self_attention_outputs[1:-1]
        # 当前层可供未来生成复用的 Key/Value 缓存元组。
        present_key_value = self_attention_outputs[-1]

        if mode=='multimodal':
            # 只有多模态模式才真正运行图文交叉注意力。
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"

            # Query 是文字，Key/Value 是图像；输出仍对应文字 token 位置。
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 图文模式下重新绑定为跨注意力后的文字向量；形状依旧 [B,T,Dt]。
            attention_output = cross_attention_outputs[0]
            # 如果请求输出注意力图，则把图文注意力图也并进返回元组。
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights                               
        # 最后逐 token 经过 FFN；chunking 可按段计算以节省峰值内存。
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 把本层最终文字表示放在返回元组第 0 位，便于下一层用 layer_outputs[0] 读取。
        outputs = (layer_output,) + outputs

        # 把本层自注意力 K/V 缓存附在最后，供增量生成。
        outputs = outputs + (present_key_value,)

        # outputs 通常是元组，第 0 项是本层主输出，后面可带注意力概率和 K/V 缓存。
        return outputs

    def feed_forward_chunk(self, attention_output):
        # FFN 第一步将每个 token 的特征从 Dt 扩到中间宽度，如 768→3072。
        intermediate_output = self.intermediate(attention_output)
        # FFN 第二步压回 Dt，并与进入 FFN 前的 attention_output 做残差。
        layer_output = self.output(intermediate_output, attention_output)
        # 返回当前 BertLayer 的 FFN 输出 [B,T,Dt] 给上层编码器。
        return layer_output


class BertEncoder(nn.Module):
    """把多层 BertLayer 串联成完整文字/图文编码器。"""
    def __init__(self, config):
        super().__init__()
        # 保存构造时的 BertConfig；forward 以后会从 self.config 读取维度、开关和默认值。
        self.config = config
        # ModuleList 会向 PyTorch 注册每层参数，层数由 JSON 配置决定。
        self.layer = nn.ModuleList([BertLayer(config,i) for i in range(config.num_hidden_layers)])
        # 默认不启用激活检查点；若训练时启用，可省显存但要重算层前向。
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multimodal',
    ):
        # 只有请求返回每层状态时才建空元组；否则用 None 避免不必要保存。
        all_hidden_states = () if output_hidden_states else None
        # 只有调用者要求看每层自注意力图，才准备元组累积它们。
        all_self_attentions = () if output_attentions else None
        # 若要求并且配置含交叉注意力，准备图文注意力图容器。
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 解码器若需逐词缓存，则准备元组收集每一层返回的 K/V。
        next_decoder_cache = () if use_cache else None
               
        # 每层接收上一层的 [B,T,D]，并输出相同形状。
        for i in range(self.config.num_hidden_layers):
            # 逐层取出独立的 BertLayer 参数。
            layer_module = self.layer[i]
            # 调用者要求看中间层时才保存本层文字状态，避免默认占用额外内存。
            if output_hidden_states:
                # 每层前或末尾把当前 [B,T,Dt] 存进元组，供分析中间层。
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 若指定按层剪/屏蔽注意力头，则取第 i 层对应 mask；否则不干预。
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 上次生成缓存按层存放；第 i 个 BertLayer 只能接自己的旧 K/V。
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 仅在训练且开启激活检查点时重算部分层，以降低显存峰值。
            if self.gradient_checkpointing and self.training:

                # 生成模式缓存当前层 K/V；普通编码模式通常不用。
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    # 关闭缓存，避免与梯度检查点冲突或在训练 labels 模式下浪费内存。
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # *inputs 将传入张量序列展开成位置参数；检查点重算时调用原层。
                        return module(*inputs, past_key_value, output_attentions)

                    # 返回内部闭包函数，供 torch.utils.checkpoint.checkpoint 在反向时再次调用。
                    return custom_forward

                # 对当前 BertLayer 使用重算技术，不在前向永久保存所有激活。
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    mode=mode,
                )
            else:
                # 常规路径：传文字表示、文字 mask、可选图像 token 与模式。
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    mode=mode,
                )

            # 这里把第 i 层处理后的文字向量重新绑定到 hidden_states，供下一层继续使用。
            hidden_states = layer_outputs[0]
            # 生成模式缓存当前层 K/V；普通编码模式通常不用。
            if use_cache:
                # 把当前层返回元组末尾的 K/V 加入总缓存；每层位置必须固定。
                next_decoder_cache += (layer_outputs[-1],)
            # 只有请求注意力图时才收集；不影响一般训练目标。
            if output_attentions:
                # 将当前层第 1 项注意力图追加到元组；仅 output_attentions=True 执行。
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 调用者要求看中间层时才保存本层文字状态，避免默认占用额外内存。
        if output_hidden_states:
            # 每层前或末尾把当前 [B,T,Dt] 存进元组，供分析中间层。
            all_hidden_states = all_hidden_states + (hidden_states,)

        # False 时返回普通元组；True 时返回可用属性名访问的 ModelOutput。
        if not return_dict:
            # 过滤 None 后按固定顺序返回元组，调用者需记住每个位置意义。
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 返回结构化对象，常访问 last_hidden_state、past_key_values 等属性。
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # dense 是全连接线性层，输入输出均为 Dt；在不同 Head 类中各有独立参数。
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Pooler 的 Tanh 激活；BLIP 常以 add_pooling_layer=False 绕过整个 Pooler。
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        # Pooler 只读取序列第 0 个 token，然后线性变换；BLIP 常设置 add_pooling_layer=False。
        pooled_output = self.dense(first_token_tensor)
        # 将首 token 线性投影后的值经 Tanh 限制在 [-1,1] 范围。
        pooled_output = self.activation(pooled_output)
        # Pooler 输出单个首 token 汇总向量；BLIP 主路径常不构造 Pooler。
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """词表预测头前的小变换：线性层、激活、归一化。"""
    def __init__(self, config):
        super().__init__()
        # dense 是全连接线性层，输入输出均为 Dt；在不同 Head 类中各有独立参数。
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 若配置的激活是名称字符串，如 gelu，就从 ACT2FN 查函数。
        if isinstance(config.hidden_act, str):
            # 词预测头变换层也根据配置选择 GELU 等激活。
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            # 若配置直接给出函数对象，则保存该对象而不从 ACT2FN 字典查字符串。
            self.transform_act_fn = config.hidden_act
        # 归一化每个 token 的特征，eps 防止除以接近零的方差。
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 词预测前的内部变换：每个 token 的 Dt 维经过线性层。
        hidden_states = self.dense(hidden_states)
        # 继续通过激活函数增加非线性，形状不变。
        hidden_states = self.transform_act_fn(hidden_states)
        # 对预测头的中间文字特征归一化，再送到词表 decoder。
        hidden_states = self.LayerNorm(hidden_states)
        # 返回本辅助层处理后的文字 token；张量形状与本层定义的输入/输出宽度对应。
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """将每个 token 的隐藏向量投影成整个词表的分数。"""
    def __init__(self, config):
        super().__init__()
        # 语言模型词预测头的前置变换层：Linear+激活+LayerNorm。
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # 输出形状从 [B,T,D] 变成 [B,T,V]，V 是词表大小。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 词表中每个候选词各有一项可训练偏置，与解码线性层输出相加。
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 先把 [B,T,Dt] 做预测头前置变换。
        hidden_states = self.transform(hidden_states)
        # 最后映射到 [B,T,V]，V 是词表大小，每词一个原始分数。
        hidden_states = self.decoder(hidden_states)
        # 返回本辅助层处理后的文字 token；张量形状与本层定义的输入/输出宽度对应。
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # BertOnlyMLMHead 的子模块，含前置变换和词表输出层。
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        # 输入最后一层文字状态 [B,T,Dt]，输出每位置词表分数 [B,T,V]。
        prediction_scores = self.predictions(sequence_output)
        # 返回 [B,T,V] 词表原始分数；V 为 tokenizer 词表大小。
        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 告诉 PreTrainedModel 这个模型使用哪种配置类，可用于 from_pretrained。
    config_class = BertConfig
    # 权重命名中主干前缀为 bert，帮助 Hugging Face 保存/加载。
    base_model_prefix = "bert"
    # 加载预训练权重时若缺少固定位置序号 buffer，可忽略此键，不代表词向量缺失。
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # LayerNorm 的比例参数设 1、偏置设 0，让初始归一化不任意缩放。
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 仅线性层确实含 bias 时才将其置零，避免访问不存在的偏置。
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    # BLIP 在同一套 BERT 层上切换 text/multimodal 模式；后者增加图文交叉注意力。
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # 保存构造时的 BertConfig；forward 以后会从 self.config 读取维度、开关和默认值。
        self.config = config

        # 把 token ID 与位置 ID 变成向量。
        self.embeddings = BertEmbeddings(config)
        
        # 多层注意力与 FFN 组成 BERT 主体。
        self.encoder = BertEncoder(config)

        # BLIP 常给 add_pooling_layer=False，因此 pooled_output 为 None，取 CLS 由调用者自己做。
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
 

    def get_input_embeddings(self):
        # 返回输入词向量层对象；resize_token_embeddings 等工具依赖此接口。
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 替换输入词向量层，常在增加 [ENC]/[DEC] 后扩词表使用。
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 逐项读取“层号→要剪的注意力头”映射，再交给对应 BertLayer。
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 3D mask 已指定“每个查询位置能读哪些位置”，只需添头维。
        if attention_mask.dim() == 3:
            # 已提供 [B,Tq,Tk] mask 时加一个头维，变为可广播到各头的 [B,1,Tq,Tk]。
            extended_attention_mask = attention_mask[:, None, :, :]
        # 普通 tokenizer 的 mask 形状 [B,T]，需要扩维才能加到 [B,H,T,T] 分数。
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                # 因果掩码防止生成第 t 个词时偷看第 t+1 个词。
                # 元组拆包：第一维是批量 B，第二维是文字长度 T。
                batch_size, seq_length = input_shape

                # 产生 0..T-1 的文字位置序号，用于构造禁止看未来的因果 mask。
                seq_ids = torch.arange(seq_length, device=device)
                # 下三角位置为 True：只能读取当前位置及其左侧文字。
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
   
                # 有历史 K/V 缓存时，当前查询长度比可读取的全部过去+当前长度短。
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    # 若有缓存，完整可读长度比新输入 T 更长；差值是历史 token 数。
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    # 把允许读取历史缓存的全 1 前缀接到当前 token 的因果下三角 mask。
                    causal_mask = torch.cat(
                        [
                            torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )                     

                # 同时满足“不是未来词”和“不是 PAD”，才允许参加注意力。
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                # 文字编码器可双向读取，只需屏蔽 PAD 位置。
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        # 有效位 1→0，无效位 0→-10000；加到注意力分数后，后者概率近 0。
        # 有效 1→0，不影响原打分；PAD 0→-10000，加到分数后 softmax 概率近 0。
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # 返回已扩到可广播维度、且无效位为大负数的注意力分数 mask。
        return extended_attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multimodal',
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        # 显式参数优先，否则使用 JSON 配置默认值。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据调用参数或 config 决定是否收集每一层中间状态。
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 显式传参优先；None 时采用模型配置默认值，决定返回对象还是普通元组。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 生成时可缓存历史 K/V；纯编码时无需缓存。
        if is_decoder:
            # 解码时是否返回每层历史 K/V；未指定时采用配置默认。
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            # 关闭缓存，避免与梯度检查点冲突或在训练 labels 模式下浪费内存。
            use_cache = False

        # input_ids 与 inputs_embeds 二选一，不能同时提供。
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 通常路径：从整数词 ID 获取 [B,T] 形状和所在设备。
        elif input_ids is not None:
            # BertModel 的 input_ids 为 [B,T] 整数词 ID；从它推断 batch 与序列长度。
            input_shape = input_ids.size()
            # 元组拆包：第一维是批量 B，第二维是文字长度 T。
            batch_size, seq_length = input_shape
            # 得到 ID 张量所在 CPU/GPU，后建的 mask 必须在相同设备。
            device = input_ids.device
        # 若上游已算好词向量 [B,T,Dt]，无需再按 ID 查 Embedding 表。
        elif inputs_embeds is not None:
            # 若已直接传词向量 [B,T,Dt]，去掉最后的特征宽度，得到 [B,T]。
            input_shape = inputs_embeds.size()[:-1]
            # 元组拆包：第一维是批量 B，第二维是文字长度 T。
            batch_size, seq_length = input_shape
            # 使用现成词向量所在 CPU/GPU，后建 mask 要在同一设备。
            device = inputs_embeds.device
        # 第三种输入路径：直接提供外部编码向量作为 BERT 层输入。
        elif encoder_embeds is not None:    
            # 去掉末尾特征维，留下 [B,T] 用来创建位置和注意力 mask。
            input_shape = encoder_embeds.size()[:-1]
            # 元组拆包：第一维是批量 B，第二维是文字长度 T。
            batch_size, seq_length = input_shape 
            # 直接提供向量时，从它读取设备位置。
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        # 生成时已经缓存的历史 token 数，用于接续位置编码与 mask。
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # 调用端未给文字有效位 mask 时，默认全部 token 都有效。
        if attention_mask is None:
            # 若有历史缓存，mask 长度必须覆盖过去 K/V 加本次新 token。
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 将 [B,T] 的 0/1 mask 扩成注意力分数能直接相加的形状。
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, 
                                                                                 device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 图像 token 只在图文融合/生成时传进来。
        if encoder_hidden_states is not None:
            # 兼容上层给多个编码器结果列表的路径；普通 BLIP 常传单个图像 token 张量。
            if type(encoder_hidden_states) == list:
                # 读取列表首个编码器输出的 batch 与序列长度；末尾宽度用 _ 忽略。
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                # 图像 token 常为 [B,P,Dv]；下划线接收暂时不关心的视觉特征宽度。
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            # 建立 [B,P] 形状，用于缺省时给每个图像 token 补全 1 mask。
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            # 多个编码器输出各有一个 mask 时，逐个转成可加到注意力分数的形式。
            if type(encoder_attention_mask) == list:
                # 列表推导式对每个 0/1 图像 mask 调用父类 invert_attention_mask。
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            # 有图像特征却未给图像 mask 时，下面默认所有图像 token 均有效。
            elif encoder_attention_mask is None:
                # 若调用者没提供图像 mask，则假设所有图像 token 都有效。
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                # 图像掩码也转为注意力可相加的形式。
                # 把图像 1/0 有效位转换成可加到图文注意力 logits 的 0/大负数形式。
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                # 把图像 1/0 有效位转换成可加到图文注意力 logits 的 0/大负数形式。
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            # 没有图像 token 传入时无需图像侧 mask，纯文字模式会跳过交叉注意力。
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # 通常从整数 token ID 取得词向量；encoder_embeds 是可选的外部向量路径。
        if encoder_embeds is None:
            # 把整数 input_ids 查词/位置向量；返回 [B,T,Dt]，是 BertEncoder 的实际输入。
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            # 若上层直接提供已算好的文字向量，就跳过整数 ID 到词向量的查表。
            embedding_output = encoder_embeds
            
        # 多层 BertLayer 根据 mode 决定只读文字，还是加图文交叉注意力。
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
        )
        # 第 0 项是最后一层全部文字 token 的特征 [B,T,D]。
        # 这是最后一层全部文字位置的表示 [B,T,Dt]，不是单个 CLS 向量。
        sequence_output = encoder_outputs[0]
        # BLIP 常禁用 pooler，因此通常取 last_hidden_state[:,0,:] 做整体表示。
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # False 时返回普通元组；True 时返回可用属性名访问的 ModelOutput。
        if not return_dict:
            # 普通元组格式：最终每 token 状态、可选 Pooler，以及缓存/中间信息。
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 结构化返回；上层通过 .last_hidden_state 获取最终 [B,T,Dt]。
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



class BertLMHeadModel(BertPreTrainedModel):
    # 在 BertModel 后接词表预测头；训练计算下一个词 loss，推理逐词生成。

    # 加载不带 Pooler 的 BLIP 变体时，旧权重多出的 pooler 键可忽略。
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 固定位置 buffer 与词预测偏置有特殊共享/初始化路径，缺失键可忽略。
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 生成任务需要所有 token 的表示，不需要单独的 pooler。
        self.bert = BertModel(config, add_pooling_layer=False)
        # 把 D 维文字特征映射成词表中每个 token 的分数。
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        # 返回语言模型词表输出 Linear 层，供 Hugging Face 通用生成/权重共享接口使用。
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # 用新输出层替换词预测线性层；例如外部需要重设词表权重时。
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,            
        is_decoder=True,
        reduction='mean',
        mode='multimodal', 
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        # 显式传参优先；None 时采用模型配置默认值，决定返回对象还是普通元组。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 有标准答案就计算训练 loss，无需缓存逐词生成状态。
        # 只有训练传入目标词 ID 时才计算交叉熵；推理没有标准答案。
        if labels is not None:
            # 关闭缓存，避免与梯度检查点冲突或在训练 labels 模式下浪费内存。
            use_cache = False

        # 内部使用因果文字 mask，并通过交叉注意力读取视觉 token。
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
        )
        
        # LM 模型取 BERT/ MED 最后一层的全部文字 token [B,T,Dt]，随后映射到词表。
        sequence_output = outputs[0]
        # [B,T,D] -> [B,T,词表大小]，每个位置都得到全词表分数。
        # cls 是语言模型词表预测头：把每个位置 Dt 维映射成词表 V 个原始分数。
        prediction_scores = self.cls(sequence_output)
        
        # 某些调用者只要下一词 logits 而不需要完整损失/结构化输出。
        if return_logits:
            # 只给出可预测下一个词的位置 0..T-2 的 logits；contiguous 整理内存。
            return prediction_scores[:, :-1, :].contiguous()  

        # 没有 labels 的纯生成调用不计算训练损失，先用 None 表示。
        lm_loss = None
        # 只有训练传入目标词 ID 时才计算交叉熵；推理没有标准答案。
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            # 位置 0...T-2 的输出应预测位置 1...T-1 的下一个词。
            # 丢最后一个预测位置，令第 t 位输出对应第 t+1 个真实词；contiguous 供后续 view 使用。
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            # 丢第一个目标 token，与上一行的预测错开一位；这才是下一词训练。
            labels = labels[:, 1:].contiguous()
            # label_smoothing=0.1 让正确答案目标稍微平滑；-100 标签自动忽略。
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1) 
            # view 把 B 和 T 合并，让交叉熵逐 token 处理。
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if reduction=='none':
                # 把逐 token 损失按样本重新分组并求和，得到 [B]；VQA 可逐答案加权。
                lm_loss = lm_loss.view(prediction_scores.size(0),-1).sum(1)               

        # False 时返回普通元组；True 时返回可用属性名访问的 ModelOutput。
        if not return_dict:
            # 无 return_dict 时组装普通元组；首项是词表分数，后续是缓存/中间信息。
            output = (prediction_scores,) + outputs[2:]
            # 有训练标签则把损失放元组首位；纯推理无损失就只返回输出。
            return ((lm_loss,) + output) if lm_loss is not None else output

        # 带属性名的生成输出对象，含 loss、logits、缓存和可选注意力图。
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        # Hugging Face 的 generate() 每生成一个 token 都会调用此函数整理输入。
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        # 调用端未给文字有效位 mask 时，默认全部 token 都有效。
        if attention_mask is None:
            # new_ones 用 input_ids 的设备/类型建全 1 mask；当前生成提示每个位置都有效。
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            # 之前的 token 已缓存 K/V，本次只输入最新一个 token。
            # 生成时前文 K/V 已存在 past 缓存，本次只送最新一个 token，避免重算前文。
            input_ids = input_ids[:, -1:]

        # 给 Hugging Face generate 返回每一步要喂 forward 的参数字典。
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        # Beam Search 保留/重排候选句时，历史 K/V 必须按相同顺序重排。
        reordered_past = ()
        # beam search 重排缓存时逐层遍历；每层自己的 Key/Value 都要同步重排。
        for layer_past in past:
            # index_select 沿 batch/beam 维按 beam_idx 重排每个 K/V，保证缓存跟候选句对齐。
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        # Beam Search 缓存已按候选新顺序重排，返回给下一次解码。
        return reordered_past
