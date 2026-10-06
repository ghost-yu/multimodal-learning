# typing 是 Python 标准库，给读代码的人和类型检查器描述数据类型，不执行张量计算。
# Dict 表示字典；Optional[T] 表示值可以是 T 或 None；Tuple 表示固定位置的多个返回值。
from typing import Dict, Optional, Tuple

# torch 是 PyTorch 主库：Tensor 是多维数组，同时可以记录求导关系。
# 此文件用 torch.ones_like 创建同形状掩码、torch.cat 拼接张量、torch.bool 表示布尔类型。
import torch
# torch.nn 是神经网络模块库，nn 是 neural network 的缩写。这里主要用 nn.Module 做类型标注。
# Module 管理参数和子模块；Module.train()/eval() 切换模式，to(device) 移动参数。
import torch.nn as nn

# 开头的点表示相对导入，从同一个 task 包读取模块，不是从网络下载。
# unwrap 意为“解开包装”：分布式训练常给原模型套一层对象，这个函数取得里面的真实模型。
from .base_task import unwrap_model
# ImageTextTask 是本仓库的图文训练任务基类；CoCaTask 继承它已有的设备处理和报告逻辑。
from .image_text_task import ImageTextTask
# cat=concatenate，padded=补齐，sequences=序列：先补齐不同长度的序列，再合成一个 batch。
# 普通 torch.cat 要求非拼接维度长度相同；这里多个微批次的文字长度可以不同。
from open_clip.utils import cat_padded_sequences


# Task 不是论文中的另一套模型，是把“模型前向”和“损失计算”接起来的工程对象。
# CoCa=Contrastive Captioner：contrastive 指图文对比，captioner 指生成图片描述。
class CoCaTask(ImageTextTask):
    """CoCa / MaMMUT training task wrapping model + CoCaLoss.

    Caption labels are AR-shifted and masked to -100 at invalid positions using the batch
    ``text_valid`` ([B, L] bool/int, True/1 = real token). Batches without a mask fall back to
    the pad-value derivation ``text != model.pad_id`` -- exact for tokenizers with a reserved pad,
    and the historical convention (which also drops genuine tokens equal to the fill id, e.g.
    SimpleTokenizer id 0) otherwise.
    """

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
            caption_loss_weight: float = 2.0,
            clip_loss_weight: float = 1.0,
            fused_caption_loss: bool = False,
            caption_z_loss_weight: float = 0.0,
            caption_loss_compute_dtype="float32",
            caption_loss_chunk_size: int = 4096,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = True,
            rank: int = 0,
            world_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        # self 是当前任务对象；super() 访问父类的方法。先让父类登记模型、设备和浮点精度。
        # device 如 cuda:0/cpu 决定在哪计算；dtype 如 float32/float16 决定每个数的表示方式。
        super().__init__(model, device=device, dtype=dtype, verbose=verbose)
        # pad-value fallback for batches without a text_valid mask (see _caption_labels)
        # pad=padding（补齐），id=词表里的整数编号。getattr(obj,name,default) 读取属性，缺失时用默认值。
        # 这里记住模型采用的补齐编号；0 只是兼容默认值，未必是专门的补齐词。
        self.pad_id = getattr(unwrap_model(model), 'pad_id', 0)
        # fused: labels are passed INTO the model forward, which computes the reduced caption CE
        # (and optional z term) via fused_linear_cross_entropy (no [B, L, V] logits materialized);
        # CoCaLoss then only applies the loss weighting. Legacy (False): model returns logits,
        # CoCaLoss computes CE.
        # fused 表示“融合”：预测词表的线性投影与 CE（cross entropy，交叉熵）一起分块计算。
        # bool(x) 转成 True/False；保存到 self 上，使稍后的 training_forward 也能读到这个选择。
        self.fused_caption_loss = bool(fused_caption_loss)
        if caption_z_loss_weight < 0:
            raise ValueError(f"caption_z_loss_weight must be non-negative, got {caption_z_loss_weight}")
        # z-loss 是本实现的可选稳定项，默认权重为 0；它不是 CoCa 原论文的第三个核心目标。
        # weight 是乘在损失前的系数，float 把传入值明确转换为浮点数。
        self.caption_z_loss_weight = float(caption_z_loss_weight)
        self.caption_loss_compute_dtype = caption_loss_compute_dtype
        self.caption_loss_chunk_size = int(caption_loss_chunk_size)
        if loss is not None:
            self.loss = loss
        elif default_loss:
            from open_clip.loss import CoCaLoss
            self.loss = CoCaLoss(
                caption_loss_weight=caption_loss_weight,
                clip_loss_weight=clip_loss_weight,
                # None 的意思是这里不再用词编号猜 padding，因为 _caption_labels 已经把无效标签改成 -100。
                pad_id=None,  # labels arrive pre-masked to -100 (built in _caption_labels)
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
                z_loss_weight=self.caption_z_loss_weight,
                compute_dtype=self.caption_loss_compute_dtype,
            )
        # else: eval-only construction, no self.loss attribute

    def _caption_labels(
            self,
            text: torch.Tensor,
            text_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """AR-shifted caption labels with invalid positions masked to -100."""
        # 优先使用显式有效位；否则才用 pad_id 猜测哪些 token 是 padding。
        # text 是整数词编号，形状 [B,L]：B=batch size（本批样本数），L=每条序列占的长度。
        # text_valid 是同形状的有效位掩码：True/1=真实词，False/0=补齐。valid 命名为“是否有效”。
        # a if 条件 else b 是条件表达式；.bool() 把 0/1 张量转为 False/True 张量。
        # 没有明确掩码时，逐元素比较 text != pad_id；真实词恰好为该编号时也会被误判，应优先传掩码。
        valid = text_valid.bool() if text_valid is not None else text != self.pad_id
        # 输入看到位置 0..T-2，目标从位置 1..T-1 开始，即预测下一个词。
        # 无效位置标为 -100，交叉熵会忽略它们。
        # [:,1:]：冒号保留所有样本，1: 去掉第 0 个词，长度变为 L-1。
        # 例：输入 [BOS,猫,坐,EOS,PAD]，目标是 [猫,坐,EOS,-100]，每个位置预测下一个词。
        # ~ 是布尔取反，选出“无效位”；masked_fill(mask,value) 在 mask 为 True 处填 value，返回新张量。
        # -100 是损失函数的 ignore_index（忽略标签），不是词表编号，不能拿去查 Embedding。
        return text[:, 1:].masked_fill(~valid[:, 1:], -100)

    def create_dummy_batch(self, *args, **kwargs):
        # dummy=模拟数据，只用于检验接口；*args 展开位置参数，**kwargs 展开关键字参数。
        # batch 是字典，如 {"image":图像张量,"text":文字编号张量}，不是一个单独 Tensor。
        batch = super().create_dummy_batch(*args, **kwargs)
        # explicit all-valid mask: dummy tokens are random, so pad-value fallback would be noise
        # ones_like(t) 创建与 t 形状和设备一致、内容全为 1 的张量；dtype=bool 使 1 变 True。
        # 模拟词全标有效，避免随机词编号 0 被当成补齐；这不代表真实训练中的 padding 都有效。
        batch["text_valid"] = torch.ones_like(batch["text"], dtype=torch.bool)
        return batch

    def _build_loss_inputs(self, model_out, batch):
        """Build CoCaLoss inputs with autoregressive shift and -100 label masking."""
        # model_out 是模型返回的字典；inputs 是准备传给损失函数的另一个字典。
        # features 是归一化全局表示 [B,E]，E 是 embedding dimension（嵌入维度），并非生成词序列。
        inputs = {
            "image_features": model_out["image_features"],
            "text_features": model_out["text_features"],
            "logit_scale": model_out["logit_scale"],
        }
        if "caption_loss_ce" in model_out:
            # fused path: the model already reduced the caption CE (labels went into forward);
            # caption_loss_z rides along when the model computed the z term
            inputs["caption_loss_ce"] = model_out["caption_loss_ce"]
            if "caption_loss_z" in model_out:
                inputs["caption_loss_z"] = model_out["caption_loss_z"]
        else:
            # 去掉最后一个输入位置的 logits，与右移后的 [1:] 标签对齐。
            # logits 是未经 softmax 的词表分数 [B,L,V]，V=vocabulary size（词表大小）。
            # :-1 保留到倒数第二项，得到 [B,L-1,V]；最后输入位置没有下一词标签，所以不计算它。
            inputs["logits"] = model_out["logits"][:, :-1]
            # get(key) 在字典缺该键时返回 None；于是 _caption_labels 可以使用 pad_id 的兼容方案。
            # labels 是监督答案编号 [B,L-1]，与 logits 的每个样本、位置一一对齐。
            inputs["labels"] = self._caption_labels(batch["text"], batch.get("text_valid"))
        return inputs

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        if self.fused_caption_loss:
            # 融合路径由模型内部直接算 caption CE，避免存完整词表 logits。
            labels = self._caption_labels(batch["text"], batch.get("text_valid"))
            model_out = self.trainable_module(
                **batch,
                labels=labels,
                caption_z_loss=self.caption_z_loss_weight != 0,
                caption_loss_compute_dtype=self.caption_loss_compute_dtype,
                caption_loss_chunk_size=self.caption_loss_chunk_size,
            )
        else:
            # **batch 把字典展开成关键字参数，例如 image=batch['image'], text=batch['text']。
            # trainable_module 由父类管理；nn.Module 的调用语法最终运行模型的 forward 方法。
            model_out = self.trainable_module(**batch)
        loss_input = self._build_loss_inputs(model_out, batch)
        # self.loss 是可调用的损失模块；output_dict=True 要求返回各项损失的命名字典。
        # 这里得到 contrastive_loss 和 caption_loss，它们是带求导关系的标量张量。
        losses = self.loss(**loss_input, output_dict=True)
        # Log the (detached) caption components on the fused path regardless of loss module; they
        # intentionally lack a ``_loss`` suffix so they are not summed into the total below.
        # 字典推导式：遍历候选键 k，只把实际存在的项加入。update 合并到原字典，属于就地修改。
        # detach() 断开求导关系，得到仅用于日志的值；数值并没有变，不能用它替代用于训练的损失。
        losses.update({k: model_out[k].detach() for k in ("caption_loss_ce", "caption_loss_z") if k in model_out})
        # 只加真正训练用的 *_loss；诊断值 caption_loss_ce 不重复相加。
        # items() 依次给出键 k 和值 v；endswith 检查字符串结尾；生成器表达式只挑训练项。
        # caption_loss_ce 不以 _loss 结尾，避免“加权 caption_loss”与原始 CE 被重复加两次。
        # total_loss 才是供外部 backward() 反向传播的总目标；本方法本身还没有更新模型权重。
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        # Report from model_out (not loss_input): _build_loss_inputs drops logit_bias, which CoCaLoss can't take
        # but we still want to log. Matches the accum path, which captures bias from inputs_no_accum before dropping.
        # 返回二元 tuple：第一个是损失字典，第二个是父类生成的报告；不是只返回 total_loss。
        return losses, self._report(model_out)

    def concat_accum_features(self, features):
        return {
            # values 是各个微批次张量组成的列表；logits 可能文字长度不同，要先补齐序列维。
            # 其余 features 通常都是 [微批大小,E]，torch.cat 默认 dim=0，直接合并样本维。
            key: cat_padded_sequences(values) if key == "logits" else torch.cat(values)
            for key, values in features.items()
        }

    def compute_accum_loss(self, inputs, inputs_no_accum, accum_batches):
        if self.fused_caption_loss:
            # The accum feature-cache replays micro-batches against cached contrastive features; the
            # fused caption term would need per-micro-batch loss accumulation weighted by valid-token
            # counts to stay exact. Not wired yet -- use the legacy (logits) path with --accum-freq.
            raise NotImplementedError(
                "fused caption loss does not support --accum-freq > 1 yet; "
                "drop --fused-caption-loss or set --accum-freq 1")
        # Mask and shift each caption before padding. The loss then averages over all valid target
        # tokens in the effective batch, regardless of each microbatch's padded length or token count.
        inputs["labels"] = cat_padded_sequences([
            self._caption_labels(b["text"], b.get("text_valid")) for b in accum_batches
        ], padding_value=-100)
        inputs["logits"] = inputs["logits"][:, :-1]
        report = self._report(inputs_no_accum)  # capture before dropping logit_bias for the loss call
        # CoCaLoss doesn't accept logit_bias
        # pop 移除字典项并返回它；默认 None 使不存在该键时不报错。
        # logit_bias 是可选分数偏置；CoCaLoss 的函数签名不接收这个参数，所以调用前去掉。
        inputs_no_accum.pop("logit_bias", None)
        losses = self.loss(
            **inputs,
            **inputs_no_accum,
            output_dict=True,
        )
        return losses, report
