"""Multimodal text generation via HuggingFace GenerationMixin.

This module requires ``transformers`` and is imported lazily by model
``generate()`` methods — ``import open_clip`` does not require transformers.
"""
# copy 是 Python 标准库；deepcopy 把配置及其内部对象复制，避免修改调用者原有生成配置。
import copy
# logging 是标准日志库；getLogger(__name__) 按当前模块名取得日志器，warning 用于输出提醒。
import logging
# Any=任意类型，Callable=可调用对象，Optional[T]=T 或 None，Tuple=多个有固定位置的返回值。
# 类型提示帮助阅读，不会自动检查词编号范围或张量维度，实际检查在下面函数里。
from typing import Any, Callable, Optional, Tuple

# torch 是 PyTorch；Tensor 是可放在 CPU/GPU 上的多维数组。这里主要处理整数词编号与视觉 token。
# full 创建指定形状且全部为某值的张量；cat 拼接；no_grad 关闭梯度记录，适用于推理。
import torch
# nn=neural network；Module 管理模型参数/缓冲张量，Linear 表示线性层，Parameter 表示可学习权重。
import torch.nn as nn
# transformers 是 Hugging Face 的模型与生成工具库，本文件借用其搜索循环，不是调用远程大模型 API。
# GenerationConfig 保存生成长度、结束词、采样和 beam search 的设置，可由用户自行传入。
from transformers import GenerationConfig
# Mixin=功能混入类，提供 generate 方法；下面的包装器要满足它要求的 forward/config 等接口。
from transformers.generation import GenerationMixin
# Causal LM=因果语言模型；OutputWithPast 是带 logits 与可选历史缓存字段的返回容器。
# 使用这个容器只是接口适配，并不意味着本实现已支持 KV cache（下方明确返回 None）。
from transformers.modeling_outputs import CausalLMOutputWithPast

_logger = logging.getLogger(__name__)

_LEGACY_SOT_TOKEN_ID = 49406
_LEGACY_EOS_TOKEN_ID = 49407
_LEGACY_PAD_TOKEN_ID = 0
# LEGACY=历史兼容默认；SOT=start of text，EOS=end of sequence，PAD=padding（补齐）。
# 这些编号只适合对应词表，不能无条件套给另一个 tokenizer；优先从实际模型属性解析。


class _SimpleConfig:
    # 名字前的 _ 表示供内部使用；SimpleConfig 是最低限度的配置接口，不创建 Transformer 参数。
    """Minimal config stub satisfying GenerationMixin's attribute access."""

    def __init__(
            self,
            vocab_size: int = 49408,
            pad_token_id: int = _LEGACY_PAD_TOKEN_ID,
            eos_token_id: int = _LEGACY_EOS_TOKEN_ID,
            bos_token_id: int = _LEGACY_SOT_TOKEN_ID,
    ):
        self.is_encoder_decoder = False
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self._attn_implementation = "eager"
        self._experts_implementation = None  # probed by transformers >= 5.x decode optimization

    def get_text_config(self, decoder=False):
        return self


class MultimodalGenerationWrapper(nn.Module, GenerationMixin):
    # Wrapper=包装器，适配已有模型的文字编码/图像交叉解码，使其能使用 HF 的 generate。
    # 它不重新训练 CoCa，图片 embedding 已由外部算好，后续每个生成步骤复用同一图像条件。
    """Thin adapter making an encode-text + cross-attend-decode pipeline
    compatible with ``GenerationMixin.generate()``.

    Created transiently by a model's ``generate()`` method.  Holds pre-computed
    image embeddings and delegates ``forward()`` to the text encoder and decoder.

    Args:
        text_encoder_fn: Callable (text_ids) -> token_embs ``(B, S, D)``.
        text_decoder_fn: Callable (image_embs, token_embs) -> logits ``(B, S, V)``.
        image_embs: Pre-computed image context for cross-attention ``(B, N, D)``.
        vocab_size: Vocabulary size for config.
        pad_token_id: Pad token id.
        eos_token_id: End-of-sequence token id.
        bos_token_id: Start-of-sequence token id.
    """
    main_input_name = "input_ids"
    _is_stateful = False

    def __init__(
            self,
            text_encoder_fn: Callable,
            text_decoder_fn: Callable,
            image_embs: torch.Tensor,
            vocab_size: int = 49408,
            pad_token_id: int = 0,
            eos_token_id: int = 49407,
            bos_token_id: int = 49406,
    ):
        super().__init__()
        # fn=function 的缩写；保存可调用函数，例如 CoCa._encode_text 的 token 输出分支。
        # 保存函数并不会马上执行，等 forward 收到文字前缀才调用。
        self._text_encoder_fn = text_encoder_fn
        self._text_decoder_fn = text_decoder_fn
        # Register as buffer so .device / expansion for beam search works.
        # register_buffer 登记无需优化器更新的 Tensor；它可以跟随模型移动设备。
        # image_embs 是 [B,Q,D] 视觉序列，不是 RGB 图片；beam search 还需要随候选扩展它。
        self.register_buffer("_image_embs", image_embs)
        self.config = _SimpleConfig(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
        self.generation_config = GenerationConfig(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    @property
    def device(self) -> torch.device:
        return self._image_embs.device

    def get_experts_implementation(self) -> dict:
        # GenerationMixin.generate() (transformers>=5.14.0) unconditionally calls this from
        # _optimize_model_for_decode() to snapshot/restore MoE routing config. It's normally
        # only defined on transformers.PreTrainedModel, which this wrapper doesn't subclass.
        # This wrapper never has MoE submodels, so there's nothing to report; an empty dict
        # makes the caller's `"grouped_mm" in experts_implementation.values()` check a no-op.
        # See https://github.com/mlfoundations/open_clip/issues/1199
        return {}

    def prepare_inputs_for_generation(
            self,
            input_ids,
            image_embs=None,
            context_valid=None,
            **kwargs,
    ):
        # TODO(kv-cache): When past_key_values is not None, slice input_ids
        # to only the last token and forward past_key_values + cache_position.
        # input_ids 为目前已知前缀 [B,S]；S 随生成增长，不是监督答案 labels。
        # inputs 是待传给 forward 的字典，图片条件用显式传入值或对象中缓存的值。
        inputs = {
            "input_ids": input_ids,
            "image_embs": image_embs if image_embs is not None else self._image_embs,
        }
        if context_valid is not None:
            inputs["context_valid"] = context_valid
        return inputs

    def forward(
            self,
            input_ids: torch.Tensor,
            image_embs: Optional[torch.Tensor] = None,
            context_valid: Optional[torch.Tensor] = None,
            **kwargs,  # absorb cache_position, attention_mask, etc.
    ) -> CausalLMOutputWithPast:
        if image_embs is None:
            image_embs = self._image_embs
        # TODO(kv-cache): Accept past_key_values, pass to decoder, return
        # updated cache.  With KV-cache, only encode the new token positions
        # and concatenate cached K/V in the self-attention layers of the
        # decoder.  The cross-attention K/V (image_embs) are constant and can
        # be cached once.
        # 每轮将“当前完整前缀”编码为 [B,S,D] token 表示；此处没有 KV cache，因而不是只算最后一个词。
        # 调用 fn 返回的是位置序列，不能拿全局 text_features [B,E] 替代。
        token_embs = self._text_encoder_fn(input_ids)
        if context_valid is not None:
            # 3-arg form only when masking is in play; 2-arg decoder fns stay compatible
            logits = self._text_decoder_fn(image_embs, token_embs, context_valid)
        else:
            logits = self._text_decoder_fn(image_embs, token_embs)
        # logits=[B,S,V]，V 为词表大小；HF 搜索取最后一个位置的分数选择下一词，再把它接到前缀。
        # past_key_values=None 表示没有返回历史 K/V 缓存，所以配置中 use_cache=False 与此保持一致。
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

    def _reorder_cache(self, past_key_values, beam_idx):
        # TODO(kv-cache): Reorder cached K/V for beam search beam reordering.
        return past_key_values


def _normalize_token_id(value: Any, allow_sequence: bool = False):
    # normalize 在这里指“统一参数类型”，不是 F.normalize 的向量长度归一化。
    # value 可为整数、单元素 Tensor 或列表；某些 tokenizer 允许多个 EOS 编号。
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        # numel=number of elements（元素总数），与 dim（维数）不同；标量和 [1] 都可能只有一个元素。
        if value.numel() == 1:
            # item() 取出单元素 Tensor 内的 Python 数；生成配置需要编号，不需要带设备的 Tensor。
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, (list, tuple)):
        # 列表推导式：遍历各 v，去掉 None，并把有效值转成 int（整数）。
        values = [int(v) for v in value if v is not None]
        if not values:
            return None
        return values if allow_sequence else values[0]
    return int(value)


def _iter_generation_sources(model: nn.Module):
    # iter=iterate（逐项遍历）；sources=可能保存词表配置的对象来源。
    # yield 每次提供一个对象，形成生成器；和 return 一次性结束整个函数不同。
    seen = set()
    for source in (
            model,
            # getattr(对象,属性名,默认值)：有文字塔就读取，没有就返回 None。
            getattr(model, 'text', None),
            getattr(model, 'text_decoder', None),
            getattr(model, 'decoder', None),
    ):
        if source is None or id(source) in seen:
            continue
        # id 是 Python 对象的身份标识，不是 token id；seen 集合防止同一配置对象被重复遍历。
        seen.add(id(source))
        yield source
        config = getattr(source, 'config', None)
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            yield config


def _resolve_from_sources(sources, names, allow_sequence: bool = False):
    for source in sources:
        for name in names:
            value = getattr(source, name, None)
            value = _normalize_token_id(value, allow_sequence=allow_sequence)
            if value is not None:
                return value
    return None


def resolve_generation_token_ids(
        model: nn.Module,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sot_token_id: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Resolve generation special tokens from explicit args, then built model/tower attrs, then legacy CLIP ids."""
    sources = tuple(_iter_generation_sources(model))
    pad_token_id = _normalize_token_id(pad_token_id)
    if pad_token_id is None:
        pad_token_id = _resolve_from_sources(sources, ('pad_id', 'pad_token_id'))
    if pad_token_id is None:
        pad_token_id = _LEGACY_PAD_TOKEN_ID

    eos_token_id = _normalize_token_id(eos_token_id, allow_sequence=True)
    if eos_token_id is None:
        eos_token_id = _resolve_from_sources(
            sources,
            ('eos_id', 'eos_token_id', 'eot_token_id', 'sep_token_id'),
            allow_sequence=True,
        )
    if eos_token_id is None:
        eos_token_id = _LEGACY_EOS_TOKEN_ID

    sot_token_id = _normalize_token_id(sot_token_id)
    if sot_token_id is None:
        sot_token_id = _resolve_from_sources(
            sources,
            ('bos_id', 'bos_token_id', 'sot_id', 'sot_token_id', 'cls_token_id'),
        )
    if sot_token_id is None:
        sot_token_id = _LEGACY_SOT_TOKEN_ID

    return pad_token_id, eos_token_id, sot_token_id


def resolve_generation_vocab_size(decoder: nn.Module) -> int:
    """Resolve the logits vocabulary size from the decoder side of a multimodal generator."""
    value = getattr(decoder, 'vocab_size', None)
    if value is not None:
        return int(value)

    lm_head = getattr(decoder, 'lm_head', None)
    if isinstance(lm_head, nn.Linear):
        return int(lm_head.out_features)
    if isinstance(lm_head, nn.Parameter):
        return int(lm_head.shape[-1])

    text_projection = getattr(decoder, 'text_projection', None)
    if isinstance(text_projection, nn.Parameter):
        return int(text_projection.shape[-1])
    if isinstance(text_projection, nn.Linear):
        return int(text_projection.out_features)

    # last resort: on some decoders (e.g. ModernMultimodalDecoder) output_dim is the contrastive
    # embed dim rather than the vocab, so only trust it once vocab_size and head introspection miss
    value = getattr(decoder, 'output_dim', None)
    if value is not None:
        return int(value)

    raise AttributeError("Could not resolve generation vocab size from decoder.")


def _right_padded(valid: torch.Tensor) -> bool:
    # right-padded=补齐全部放在右侧。有效位 [True,True,False] 合法，[True,False,True] 不合法。
    # ~valid 把补齐位变 True；cummax 是沿序列累计最大值，第一次补齐之后都标 True。
    seen_pad = (~valid).cummax(dim=1).values
    # 如果“已经出现补齐”和“当前仍是真词”同时成立，说明补齐后又出现真实词，违反右补齐规则。
    # any() 判断是否至少一个 True；bool 转 Python 布尔值；not 把“有违规”转成“符合规则”。
    return not bool((seen_pad & valid).any())


def prepare_generation_prompt(
        text: Optional[torch.Tensor],
        text_valid: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        sot_token_id: int,
) -> Tuple[torch.Tensor, bool]:
    """Create/validate the common-length prompt consumed by GenerationMixin."""
    squeeze_output = False
    if text is None:
        # 没有给文字提示时，每张图片以一个 SOT/BOS 词开始；full 生成 [B,1] 的整数编号张量。
        # long 是 int64，Embedding 查表需要整数而不是浮点；返回第二项用于记录是否恢复一维输出。
        return torch.full((batch_size, 1), sot_token_id, device=device, dtype=torch.long), squeeze_output

    if text.dim() == 1:
        # unsqueeze(0) 加一个批次维：[S] → [1,S]，不改变词编号顺序。
        text = text.unsqueeze(0)
        if text_valid is not None:
            text_valid = text_valid.unsqueeze(0)
        squeeze_output = batch_size == 1
    if text.dim() != 2:
        raise ValueError(f"generation prompt must be a 1D or 2D token tensor, got shape {tuple(text.shape)}.")
    if text.shape[0] != batch_size:
        raise ValueError(
            f"generation prompt batch ({text.shape[0]}) must match image batch ({batch_size})."
        )

    # to 同时指定设备和数据类型，让提示词与图像条件在同一设备计算，整数化为词表查询准备。
    text = text.to(device=device, dtype=torch.long)
    if text_valid is not None:
        text_valid = text_valid.to(device=device, dtype=torch.bool)
        if text_valid.shape != text.shape:
            raise ValueError(
                f"text_valid shape {tuple(text_valid.shape)} must match text shape {tuple(text.shape)}."
            )
        if not _right_padded(text_valid):
            raise ValueError("generation text_valid must describe right-padded prompts.")
        if text_valid.any():
            # any(dim=0) 在批次维上汇总，找至少一个样本有真实词的位置。
            # nonzero() 返回这些位置的索引；[-1] 取最后一个；索引从 0 开始，所以 +1 变保留长度。
            keep_len = int(text_valid.any(dim=0).nonzero()[-1].item()) + 1
        else:
            keep_len = 1
        # 删去所有样本共同的右侧补齐；这里是切片，不是把词内容截断到用户想要的输出长度。
        text = text[:, :keep_len]
        text_valid = text_valid[:, :keep_len]
        if not bool(text_valid.all()):
            raise ValueError(
                "Generation prompts with text_valid must have the same valid length in every row; "
                "pass same-length unpadded prompts or generate each prompt length separately."
            )

    return text, squeeze_output


def build_generation_config(
        generation_config: Optional[GenerationConfig],
        generation_type: str,
        seq_len: int,
        min_seq_len: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        num_beam_groups: int,
        repetition_penalty: float,
        pad_token_id: int,
        eos_token_id: int,
        sot_token_id: int,
) -> GenerationConfig:
    if generation_config is not None:
        # 深复制后只改副本，避免多次生成时用户配置被悄悄修改。
        generation_config = copy.deepcopy(generation_config)
        if getattr(generation_config, 'pad_token_id', None) is None:
            generation_config.pad_token_id = pad_token_id
        if getattr(generation_config, 'eos_token_id', None) is None:
            generation_config.eos_token_id = eos_token_id
        if getattr(generation_config, 'bos_token_id', None) is None:
            generation_config.bos_token_id = sot_token_id
        generation_config.use_cache = False
        return generation_config

    if seq_len <= min_seq_len:
        raise ValueError("seq_len must be larger than min_seq_len.")

    gen_kwargs = dict(
        # max_length 是“提示词+新生成词”的总长度；不是只允许生成 seq_len 个新词。
        max_length=seq_len,
        min_length=min_seq_len,
        repetition_penalty=repetition_penalty,
        bos_token_id=sot_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
    )
    if generation_type == "beam_search":
        if num_beam_groups > 1:
            _logger.warning(
                "Group beam search (num_beam_groups > 1) requires the transformers community "
                "extension. Falling back to standard beam search (num_beam_groups=1). Pass a "
                "GenerationConfig directly for full control."
            )
            num_beam_groups = 1
        gen_kwargs.update(num_beams=num_beams, num_beam_groups=num_beam_groups)
    elif generation_type == "top_p":
        # top_p 是 nucleus sampling（核采样）：从累计概率达到阈值的候选集合中采样。
        # do_sample=True 允许随机选择，temperature 调节词概率尖锐程度；不是对比学习中的 logit_scale。
        gen_kwargs.update(do_sample=True, top_p=top_p, temperature=temperature)
    elif generation_type == "top_k":
        # top_k 只保留分数最高的 k 个候选词再采样；k=1 时实际上只剩最高分词。
        gen_kwargs.update(do_sample=True, top_k=top_k, temperature=temperature)
    else:
        raise ValueError(
            f"generation_type must be one of 'beam_search', 'top_p', 'top_k', got {generation_type!r}"
        )
    return GenerationConfig(**gen_kwargs)


def validate_generation_lengths(
        prompt_len: int,
        generation_config: GenerationConfig,
        seq_len: int,
        max_seq_len: Optional[int],
        context_length: Optional[int],
) -> int:
    max_new_tokens = getattr(generation_config, 'max_new_tokens', None)
    if max_new_tokens is not None:
        # 若指定 max_new_tokens（新增词上限），必须加上已有前缀长度才能与模型 context length 比较。
        max_length = prompt_len + int(max_new_tokens)
    else:
        max_length = getattr(generation_config, 'max_length', None) or seq_len
    if max_seq_len is not None and max_length > max_seq_len:
        raise ValueError(f"generation max_length ({max_length}) cannot exceed max_seq_len ({max_seq_len}).")
    if context_length is not None and context_length > 0 and max_length > context_length:
        raise ValueError(
            f"generation max_length ({max_length}) cannot exceed decoder context_length ({context_length})."
        )
    if prompt_len > max_length:
        raise ValueError(f"generation prompt length ({prompt_len}) cannot exceed max_length ({max_length}).")
    return max_length


def generate_multimodal(
        model: nn.Module,
        image: torch.Tensor,
        image_embs_fn: Callable[[torch.Tensor], torch.Tensor],
        text_encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        text_decoder_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        decoder: nn.Module,
        text: Optional[torch.Tensor] = None,
        text_valid: Optional[torch.Tensor] = None,
        seq_len: int = 30,
        max_seq_len: int = 77,
        temperature: float = 1.,
        generation_type: str = "beam_search",
        top_p: float = 0.1,
        top_k: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sot_token_id: Optional[int] = None,
        num_beams: int = 6,
        num_beam_groups: int = 3,
        min_seq_len: int = 5,
        stopping_criteria=None,
        repetition_penalty: float = 1.0,
        fixed_output_length: bool = False,
        generation_config: Optional[GenerationConfig] = None,
) -> torch.Tensor:
    if stopping_criteria is not None:
        import warnings
        warnings.warn(
            "stopping_criteria is deprecated and ignored. Use "
            "generation_config=GenerationConfig(...) for full control.",
            DeprecationWarning,
            stacklevel=2,
        )

    pad_token_id, eos_token_id, sot_token_id = resolve_generation_token_ids(
        model,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        sot_token_id=sot_token_id,
    )
    generation_config = build_generation_config(
        generation_config=generation_config,
        generation_type=generation_type,
        seq_len=seq_len,
        min_seq_len=min_seq_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        sot_token_id=sot_token_id,
    )

    # 先记住原来的 train/eval 模式，生成结束后恢复。eval 影响 Dropout 等模块，不自动关闭梯度。
    was_training = model.training
    model.eval()
    try:
        # with 创建上下文范围；no_grad 关闭范围内的自动求导记录，减少生成时的显存/内存开销。
        with torch.no_grad():
            # image_embs_fn may return either embs or (embs, context_valid) -- NaFlex token-mode
            # towers surface patch validity so padded K/V get masked in cross-attention
            # 图片只编码一次，函数由 CoCa._generation_components 提供；后续各生成步复用这些视觉 token。
            image_embs = image_embs_fn(image)
            context_valid = None
            if isinstance(image_embs, tuple):
                image_embs, context_valid = image_embs
            prompt, squeeze_output = prepare_generation_prompt(
                text=text,
                text_valid=text_valid,
                batch_size=image_embs.shape[0],
                device=image_embs.device,
                sot_token_id=sot_token_id,
            )
            target_len = validate_generation_lengths(
                prompt_len=prompt.shape[1],
                generation_config=generation_config,
                seq_len=seq_len,
                max_seq_len=max_seq_len,
                context_length=getattr(decoder, 'context_length', None),
            )

            # 创建轻量接口包装，不是创建第二个随机初始化的 CoCa。编码/解码函数仍来自原模型。
            wrapper = MultimodalGenerationWrapper(
                text_encoder_fn=text_encoder_fn,
                text_decoder_fn=text_decoder_fn,
                image_embs=image_embs,
                vocab_size=resolve_generation_vocab_size(decoder),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                bos_token_id=sot_token_id,
            )
            generate_kwargs = dict(image_embs=image_embs)
            if context_valid is not None:
                # passed as a model kwarg (not a closure) so HF beam search expands it
                # alongside image_embs via _expand_inputs_for_generation
                generate_kwargs['context_valid'] = context_valid
            # generate 是继承 HF GenerationMixin 的搜索循环：重复 forward、挑下一词、追加前缀。
            # 返回 output 是词编号，不是已经 decode 成中文/英文字符串。
            output = wrapper.generate(
                prompt,
                generation_config=generation_config,
                **generate_kwargs,
            )

            if fixed_output_length and output.shape[1] < target_len:
                # 如果提前遇到 EOS，按要求把返回长度补到 target_len。PAD 是返回格式补齐，不是继续生成内容。
                pad_len = target_len - output.shape[1]
                output = torch.cat(
                    (output, torch.full(
                        (output.shape[0], pad_len), pad_token_id,
                        device=image_embs.device, dtype=output.dtype,
                    )),
                    dim=1,
                )

            if squeeze_output:
                # squeeze(0) 删除大小为 1 的批次维，恢复调用者最初传一维提示时的输出样式。
                output = output.squeeze(0)
            return output
    finally:
        # finally 不论正常返回还是抛异常都会执行；train(True/False) 恢复调用前模式，避免生成改变后续训练行为。
        model.train(was_training)
