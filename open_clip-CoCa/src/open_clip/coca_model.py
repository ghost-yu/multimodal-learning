# typing 属于 Python 标准库：Dict=字典类型，List=列表，Optional[T]=T 或 None，Union[A,B]=两种类型之一。
# 类型提示不创建数据，也不自动把数据转成目标类型；实际 Tensor 计算依赖下面的 torch。
from typing import Dict, List, Optional, Union

# PyTorch 主库 torch 提供 Tensor、多维数组运算和自动求导；此处用 ones、no_grad、float16 等。
import torch
# nn=neural network，模块库；Module 管理子网络，Parameter 登记可学习参数，Identity 原样返回输入。
from torch import nn
# functional 是以普通函数提供的神经网络运算，F 是惯例简称；F.normalize 在此把向量长度归一化为 1。
from torch.nn import functional as F
# NumPy 是数组与数值计算库；np 是约定简称。这里只用 np.log 求自然对数初始化温度，未用它训练。
import numpy as np
# dataclasses 是标准库；@dataclass 根据类中的字段生成 __init__ 等方法，适合保存配置。
from dataclasses import dataclass

# . 表示同包内导入；traits=模型特性，供工厂/训练代码识别模型具备哪些功能。
from .model_traits import COCA_TRAITS
from .transformer import (
    AttentionalPooler,
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    ModernMultimodalTransformer,
    MultimodalTransformer,
)
# fused=融合，caption=图片描述：将解码器隐藏向量投到词表并计算 CE，避免保存整个 [B,L,V] logits。
# CE=cross entropy（交叉熵）；这个优化属于本地实现，不是论文中新增的学习目标。
from .loss import fused_caption_loss
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower, _forward_tower_intermediates


# 装饰器 @dataclass 把下方类转成配置容器；CLIPTextCfg 的字段也由继承保留。
@dataclass
class MultimodalCfg(CLIPTextCfg):
    # Cfg 是 configuration（配置）的缩写；multimodal=多模态，这些字段描述读取图片的文字解码器。
    # MLP=multi-layer perceptron（多层感知机）；ratio=比例，隐藏层宽度常为模型宽度乘该值。
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    # MaMMUT decoder (MultimodalDecoder) fields, ignored by the CoCa decoder builder
    cross_attn_ratio: int = 1  # one cross-attn block per N self-attn layers (2 -> after layers 0, 2, 4, ...)
    use_pad_mask: bool = True  # mask pad tokens from attention in the bi-directional contrastive pass (legacy: False)
    pool_type: str = 'avg'  # contrastive pool: 'avg' masked mean excl pads | 'avg_all' mean incl pads (legacy)
    proj_type: str = 'none'  # contrastive text projection; 'none' (paper/legacy) requires embed_dim == width
    tie_lm_head: bool = False  # share lm_head weight with token_embedding (modern decoder only)


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    # NOTE: CoCa passes the vocab size as ``embed_dim`` -- the decoder's output projection is the vocab head.
    # isinstance(x,dict) 判断是否字典；**字典把键值展开为构造参数，转换成可用 .width 访问的配置对象。
    # 此函数的 embed_dim 在 CoCa 调用时是词表大小 V，不是对比向量宽度 E，注意同名参数在不同函数的含义。
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg

    if multimodal_cfg.text_arch == 'modern':
        # modern decoder is cfg-driven; act/norm come from mlp_type / norm_type (quick_gelu N/A)
        return ModernMultimodalTransformer(multimodal_cfg, vocab_size=embed_dim)

    # act=activation（激活函数）；选择的是类，后面网络构造时实例化。GELU 为非线性变换，QuickGELU 为近似版本。
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        # output_dim 是每个文字位置最终预测的分数个数，在这里等于 V=词表大小。
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class MultimodalGenerationMixin:
    """Caption generation for models supplying image, text, and decoder components."""

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,
        top_k=1,
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False,
        generation_config=None,
        text_valid=None,
    ):
        try:
            from .generation import generate_multimodal
        except Exception as e:
            raise RuntimeError(
                "Please install transformers for generate functionality. "
                "`pip install transformers`."
            ) from e

        return generate_multimodal(
            self,
            image=image,
            **self._generation_components(),
            text=text,
            text_valid=text_valid,
            seq_len=seq_len,
            max_seq_len=max_seq_len,
            temperature=temperature,
            generation_type=generation_type,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            sot_token_id=sot_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            min_seq_len=min_seq_len,
            stopping_criteria=stopping_criteria,
            repetition_penalty=repetition_penalty,
            fixed_output_length=fixed_output_length,
            generation_config=generation_config,
        )


# 多继承：Mixin（功能混入类）提供 generate；nn.Module 提供参数登记、to、train/eval、自动调用 forward。
# CoCa 名字来自 Contrastive Captioner，联合学习“图文配对”和“根据图片续写文字”。
class CoCa(MultimodalGenerationMixin, nn.Module):
    traits = COCA_TRAITS

    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # embed=embedding（嵌入），dim=dimension（维度）；这里 E 是图文全局向量的共同宽度。
        # self.属性 会保留在这个模型对象里，其他方法可以访问；局部变量不会自动成为对象属性。
        self.embed_dim = embed_dim
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        if vision_cfg.timm_model_name:
            # timm towers are supported in token mode only, with model-level attentional pooling:
            # the tower returns {'pooled', 'patch_tokens', 'patch_valid'} and the paper poolers
            # (generative n_queries + contrastive) live on the model, pooling the trunk tokens with
            # validity masking. Masking terminates at the pooler: the caption decoder cross-attends
            # the fixed-count pooled queries, so no context_valid threading is needed downstream
            # (contrast MaMMUT, which cross-attends raw patch tokens and threads image_embs_valid).
            if not vision_cfg.output_tokens:
                raise ValueError(
                    "CoCa with a timm vision tower requires vision_cfg.output_tokens=true "
                    "(token mode) so model-level attentional pooling can consume trunk tokens.")
            if vision_cfg.attentional_pool not in ('cascade', 'parallel'):
                raise ValueError(
                    "CoCa with a timm vision tower requires vision_cfg.attentional_pool "
                    "'cascade' or 'parallel' (paper pooling); the caption decoder consumes the "
                    "pooled queries, not raw patch tokens.")
        if getattr(text_cfg, 'text_arch', None) == 'modern' and \
                getattr(text_cfg, 'attention_mode', None) == 'bidirectional':
            # The caption decoder consumes the tower's contextualized token embeddings; under
            # bidirectional attention position i carries future-token information the decoder's
            # causal mask cannot undo, so the caption loss can copy and generation would mismatch
            # training. Fail fast rather than train a silently-broken caption objective. (Masked
            # 'mean' pooling for the contrastive readout works fine over a causal tower.)
            raise ValueError(
                "CoCa does not support a bidirectional modern text tower "
                "(text_cfg.attention_mode='bidirectional'): its token embeddings leak future "
                "tokens into the caption decoder. Use attention_mode='causal'.")

        # 文本塔一方面产出整句向量供对比，一方面保留 token 供生成。
        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        # 视觉塔同样产出全局向量与图像 token，两者用途不同。
        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        if vision_cfg.timm_model_name:
            # model-level paper poolers over trunk tokens (see the token-mode note above)
            norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
            trunk_dim = self.visual.trunk.num_features
            self.attn_pool_type = vision_cfg.attentional_pool
            self.pool_norm = norm_layer(trunk_dim)  # pre-pool norm, mirrors native ln_post-on-width
            self.attn_pool = AttentionalPooler(
                embed_dim,
                trunk_dim,
                n_head=vision_cfg.attn_pooler_heads,
                n_queries=vision_cfg.attn_pooler_queries,
            )
            self.attn_pool_contrastive = AttentionalPooler(
                embed_dim,
                embed_dim if self.attn_pool_type == 'cascade' else trunk_dim,
                n_head=vision_cfg.attn_pooler_heads,
                n_queries=1,
            )
            with torch.no_grad():  # match VisionTransformer.init_parameters pooler query init
                nn.init.normal_(self.attn_pool.query, std=embed_dim ** -0.5)
                nn.init.normal_(self.attn_pool_contrastive.query, std=embed_dim ** -0.5)
            # Paper pooling replaces the tower's pooled readout entirely; its parameters
            # (trunk fc_norm + TimmModel head proj) would never receive grad and trip DDP's
            # unused-parameter check. Remove them rather than carry dead weights. (MaMMUT keeps
            # them: it uses the tower's pooled output as the contrastive latent.)
            self.visual.head = nn.Identity()
            if getattr(self.visual.trunk, 'fc_norm', None) is not None:
                self.visual.trunk.fc_norm = nn.Identity()
        else:
            self.attn_pool_type = ''
            self.pool_norm = None
            self.attn_pool = None
            self.attn_pool_contrastive = None

        # 多模态文本解码器读取图像 token 和文字前缀，预测下一个词。
        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        # lshape=logit-scale shape（尺度参数的形状）；[] 创建零维标量，[1] 创建长度 1 的向量。
        # logit 指未经 softmax 的打分；scale 是缩放系数，增强正确与错误配对的分数差距。
        lshape = [1] if nonscalar_logit_scale else []
        # torch.ones(lshape) 产生全 1 张量；Parameter 将它登记为可学习权重，优化器会更新。
        # 保存的是 log(1/τ) 而不是 1/τ 本身，τ=tau（温度），初始化为 0.07。
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

        # pad id is derived from the text tower (the id it masks with: text_cfg.pad_id for native
        # towers, the transformers config pad_token_id for HF towers) so loss ignore_index and
        # generation defaults stay consistent with tower masking and tokenizer padding. The 0
        # fallback is the historical CLIP fill convention (SimpleTokenizer reserves no pad token;
        # 0 is a real vocab token).
        # getattr(obj,name,default) 查对象属性，没有时返回默认值。pad=padding（为了对齐长度添加的补齐）。
        # id 是整数编号；BOS=beginning of sequence（序列开始），EOS=end of sequence（序列结束）。
        pad_id = getattr(self.text, 'pad_id', None)
        self.pad_id = 0 if pad_id is None else int(pad_id)
        self.bos_id = getattr(self.text, 'bos_id', None)
        self.eos_id = getattr(self.text, 'eos_id', None)

        self.context_length = multimodal_cfg.context_length

    def set_grad_checkpointing(self, enable: bool = True, impl: str = 'inline'):
        self.visual.set_grad_checkpointing(enable, impl=impl)
        self.text.set_grad_checkpointing(enable, impl=impl)
        self.text_decoder.set_grad_checkpointing(enable, impl=impl)

    def _encode_image(self, images, normalize: bool = True):
        # images 通常为浮点 [B,3,H,W]：批次、RGB 通道、高、宽；self.visual 是 __init__ 创建的视觉塔。
        # nn.Module 的 obj(x) 会调用 forward；out=output，可能是字典，也可能是二元 tuple，由视觉配置决定。
        out = self.visual(images)
        if isinstance(out, dict):
            # timm token mode: {'pooled', 'patch_tokens', 'patch_valid'} -- masked model-level
            # pooling; the tower's own 'pooled' readout is unused (paper pooling replaces it).
            # patch=图像小块；patch_tokens 为 [B,P,Dv]，P=小块数，Dv=视觉主干宽度。
            # LayerNorm 在特征维归一化，仍保留 [B,P,Dv]；不是把所有图片混在一起算 batch 均值。
            tokens = self.pool_norm(out['patch_tokens'])
            patch_valid = out['patch_valid']
            # attn=attention，pool=pooling（聚合）；可学习查询从全部视觉小块读取信息。
            # tokens_embs=多个 token 的 embeddings，得到 [B,Q,E]；Q 是池化查询数量，并非原图小块数。
            # key_valid 使无效的小块不能被注意力读取；不是语言损失里的 -100 标签。
            tokens_embs = self.attn_pool(tokens, key_valid=patch_valid)
            if self.attn_pool_type == 'cascade':
                # pooled queries are fixed-count and all valid -> no mask from here on
                # cascade=串联：对比池化器读取“生成池化器的输出”，不是直接读取视觉主干。
                # 单查询输出 [B,1,E]；[:,0] 取查询维第 0 项，得到整图 [B,E]。
                # latent=隐表示，不是类别编号；image_latent 后面用来算图文相似度。
                image_latent = self.attn_pool_contrastive(tokens_embs)[:, 0]
            else:  # parallel: contrastive pooler reads the (masked) trunk tokens directly
                image_latent = self.attn_pool_contrastive(tokens, key_valid=patch_valid)[:, 0]
        else:
            image_latent, tokens_embs = out
        # normalize 默认 True；dim=-1 表示最后一维 E，把每张图的向量除以自身 L2 长度。
        # 例 [3,4] 除以 5 得 [0.6,0.8]；图文都单位化后，点积就是余弦相似度。
        # 只归一化全局 image_latent，生成用的 tokens_embs 在这行没有被同样处理。
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, text_valid=None, normalize: bool = True):
        # text towers keep the HF-style attention_mask kwarg (single-sequence scope); the parent
        # multimodal interface names the mask by modality (text_valid, alongside NaFlex patch_valid)
        text_latent, token_emb = self.text(text, attention_mask=text_valid)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, text_valid=None, normalize: bool = True):
        """Encode text, optionally using an exact validity mask.

        ``text_valid`` was inserted before ``normalize``. Legacy positional calls such as
        ``encode_text(text, False)`` must use ``normalize=False`` after this breaking API change.
        """
        text_latent, _ = self._encode_text(text, text_valid=text_valid, normalize=normalize)
        return text_latent

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            text_valid: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Breaking positional API note: ``text_valid`` was inserted after ``text``; callers passing
        ``image_indices`` or later arguments positionally must switch those arguments to keywords.

        Args:
            image: Input image tensor
            text: Input text tensor
            text_valid: Optional [B, L] bool/int text validity (True/1 = real token); pad-value fallback when absent
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize: L2 Normalize final image and text features (if present)
            normalize_intermediates: Apply final encoder norm layer to all intermediates (if possible)
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert False, 'FIXME, needs implementing'

        if image is not None:
            output.update(_forward_tower_intermediates(
                self.visual, image, 'image_features', normalize,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            ))

        if text is not None:
            output.update(_forward_tower_intermediates(
                self.text, text, 'text_features', normalize,
                attention_mask=text_valid,
                indices=text_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=text_output_fmt,
                output_extra_tokens=text_output_extra_tokens,
            ))

        # FIXME text decoder
        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None
        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            text_valid: Optional[torch.Tensor] = None,
            image_latent: Optional[torch.Tensor] = None,
            image_embs: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            caption_z_loss: bool = False,
            caption_loss_compute_dtype=torch.float32,
            caption_loss_chunk_size: int = 4096,
    ):
        """text_valid: optional [B, L] bool/int text validity (True/1 = real token), consumed by the
        text tower's pad/cls masking (passed down as its HF-style ``attention_mask``); validity falls
        back to ``text != pad_id`` when absent. Caption logits are causal over right-padded text and
        need no mask; label masking for the caption loss happens task-side.

        Breaking positional API note: ``text_valid`` was inserted after ``text``; callers passing
        ``image_latent`` or later arguments positionally must switch those arguments to keywords.

        labels: optional [B, L-1] AR-shifted caption labels (-100 = ignore, task-built). When given,
        returns the reduced caption CE as ``caption_loss_ce`` (plus ``caption_loss_z`` when
        ``caption_z_loss`` is set) via the fused linear cross-entropy (full-vocab logits are
        never materialized) instead of ``logits``. Loss weighting is applied downstream (CoCaLoss)."""
        # image_latent: [B,E] 整图向量；image_embs: [B,P,D] 图像 token。
        # 若调用者已传入编码结果，则不重复计算视觉塔。
        if image is not None and (image_latent is None or image_embs is None):
            # 二元解包：第一个返回值 [B,E] 是对比全局向量，第二个 [B,Q,E] 是生成条件的视觉 token。
            # image_embs 的 embs=embeddings 复数；它不是原始 RGB 像素，也不是全局向量。
            image_latent, image_embs = self._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        # text_latent 用于对比，token_embs 用于逐词生成。
        # text 是整数 [B,L]，首先查 Embedding 表，再经过只看过去的文字 Transformer。
        # text_latent 为 [B,E] 整句表示；token_embs 为每位置的上下文表示，供上半段多模态解码器读取。
        # text_valid 是文字有效位 [B,L]，不要和监督 labels [B,L-1] 混淆。
        text_latent, token_embs = self._encode_text(text, text_valid=text_valid)

        if image_latent is None:
            return {"text_features": text_latent}

        # 三项是后续 CLIP 式双向对比损失所需的输入。
        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            # exp() 求自然指数，把存储的 log(1/τ) 还原成正数 1/τ；论文相似度 s/τ 在这里乘 scale。
            "logit_scale": self.logit_scale.exp(),
        }
        # labels 存在时直接算 caption CE，可避免创建很大的 [B,T,词表大小] logits。
        if labels is not None:
            # 图像 token 是条件，文字 token 是当前已知前缀。
            # hidden=hidden states（隐藏状态），[B,L,D]；return_hidden=True 暂时不投影到巨大的词表 V。
            # text_decoder 即 __init__ 中创建的多模态 Transformer，文字自注意力有因果掩码，图片交叉注意力没有词序限制。
            hidden = self.text_decoder(image_embs, token_embs, return_hidden=True)
            caption_losses = fused_caption_loss(
                # LM=language model（语言模型），head=输出头；参数含词表投影权重和可选 bias。
                # 单星号 * 展开二元 tuple 为两个位置参数，不是乘法；labels 已由训练任务右移并屏蔽补齐。
                hidden, labels, *self.text_decoder.lm_head_params,
                chunk_size=caption_loss_chunk_size,
                z_loss=caption_z_loss,
                compute_dtype=caption_loss_compute_dtype,
            )
            # update 将损失字典合进输出；输出的是未乘 caption_loss_weight 的 CE，稍后由 CoCaLoss 加权。
            out_dict.update(caption_losses)
        else:
            # 未给 labels 时返回各词位置的词表分数，供外部算损失或分析。
            # 默认返回 [B,L,V] 词表原始分数；并未取 argmax 成词编号，也未自动右移 labels。
            # 当前实现把输入/目标错位处理放在 CoCaTask，单独调用模型时也必须自己保持这个对齐关系。
            out_dict["logits"] = self.text_decoder(image_embs, token_embs)
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict

    def _generation_components(self):
        return dict(
            image_embs_fn=lambda images: self._encode_image(images)[1],
            text_encoder_fn=lambda ids: self._encode_text(ids)[1],
            text_decoder_fn=self.text_decoder,
            decoder=self.text_decoder,
        )
