# torch 构建标签和保存检索分数，nn 定义任务层，F 提供损失与 softmax。
import torch
import torch.nn as nn
import torch.nn.functional as F
# 这些标准库负责评估阶段的文件查找、结果序列化和缓存。
import os
import glob
import json
# tqdm 显示大检索集评估进度。
import tqdm
# functools 缓存重复计算的辅助结果。
import functools
# dist 在多 GPU 间汇总图文向量，增加对比学习负例。
import torch.distributed as dist

# 测试集评估使用分布式采样器将样本分给不同 GPU。
from torch.utils.data.distributed import DistributedSampler
# rearrange 以直观的维度表达式重排多维张量。
from einops import rearrange
from pytorch_lightning.utilities.distributed import rank_zero_info

# 自定义 all_gather 用于分布式评估中的张量汇总。
from vlmo.modules.dist_utils import all_gather


def compute_mlm(pl_module, batch):
    """图文 MLM：图片也作为上下文，预测文字里被遮住的词。"""
    # infer() 将图文拼在一起编码；mask_text=True 选择遮词版输入。
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    # 每个文字位置输出整个词表的分数 [B,T,V]。
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    # -100 表示无需预测的位置，其他位置是正确词 ID。
    mlm_labels = infer["text_labels"]

    # view(-1,V) 把 B 和 T 合并，按词位置计算分类交叉熵。
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        # 跳过未遮盖的词以及不需要计算损失的位置。
        ignore_index=-100,
    )

    # 返回损失和可供日志/准确率计算的预测与标签。
    ret = {
        # 原版给图文 MLM 乘 0.25，避免它相对其他目标权重过大。
        "mlm_loss": mlm_loss * 0.25,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    # 根据模型模式选择训练指标或验证指标对象。
    phase = "train" if pl_module.training else "val"
    # getattr 动态取得 train_mlm_loss 或 val_mlm_loss。
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    # 统计被遮盖词的预测正确率。
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_textonly_mlm(pl_module, batch):
    """纯文字 MLM：用于 VLMo 的语言预训练阶段。"""
    # 与上面不同，这里没有输入任何图片 token。
    infer = pl_module.infer_text_mlm(batch, mask_text=True)
    # 每个文字位置预测一个词表分布。
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    # 与图文 MLM 同样忽略标签为 -100 的位置；这里不乘 0.25。
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    # 指标名称前缀是 textmlm，避免与图文 MLM 混淆。
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_textmlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_textmlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"textmlm/{phase}/loss", loss)
    pl_module.log(f"textmlm/{phase}/accuracy", acc)

    return ret


def compute_itm_hardneg(pl_module, batch, sim_i2t, sim_t2i):
    """ITM：用 ITC 相似度挑难负例，再判断图文是否匹配。"""
    # 正例、负图、负文三组各含 B 个样本。
    pos_len = batch["text_ids"].size(0)
    neg_len = batch["text_ids"].size(0)
    bsz = batch["text_ids"].size(0)
    # 正例标签 1；两类错误配对标签 0，最终长度 3B。
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    # 复制字典壳，避免后面拼负例时直接改输入 batch。
    batch = {k: v for k, v in batch.items()}
    # 首先融合 B 对正确图文，得到正例的整体 CLS 向量。
    infer_pos = pl_module.infer(batch, mask_text=False, mask_image=False)

    batch_text_ids = infer_pos["text_ids"]
    batch_text_masks = infer_pos["text_masks"]
    batch_image = infer_pos["image"]

    # 采样负例无需梯度；多卡汇总只是扩充候选池。
    with torch.no_grad():
        # world_size 是 GPU 进程数，rank 是当前进程编号。
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        # 每张 GPU 预留接收其他 GPU 张量的缓冲区。
        gathered_text_ids = [
            torch.zeros_like(batch_text_ids) for _ in range(world_size)
        ]
        gathered_text_masks = [
            torch.zeros_like(batch_text_masks) for _ in range(world_size)
        ]
        gathered_image = [
            torch.zeros_like(batch_image) for _ in range(world_size)
        ]

        # 所有 GPU 交换文字 ID、文字有效位和图片张量。
        dist.all_gather(gathered_text_ids, batch_text_ids)
        dist.all_gather(gathered_text_masks, batch_text_masks)
        dist.all_gather(gathered_image, batch_image)

        # 把本卡样本放最前面，再拼其他卡样本；对角线仍是本卡真配对。
        all_text_ids = torch.cat(
            [batch_text_ids]
            + gathered_text_ids[:rank]
            + gathered_text_ids[rank + 1 :]
        )
        all_text_masks = torch.cat(
            [batch_text_masks]
            + gathered_text_masks[:rank]
            + gathered_text_masks[rank + 1 :]
        )
        all_image = torch.cat(
            [batch_image]
            + gathered_image[:rank]
            + gathered_image[rank + 1 :]
        )

    # ITC 分数转成抽样概率；相似度越高，越可能成为“难负例”。
    with torch.no_grad():       
        weights_i2t = F.softmax(sim_i2t[:bsz, :].float(), dim=1)
        weights_t2i = F.softmax(sim_t2i[:bsz, :].float(), dim=1)

        # 自己的正确配对不能被抽作负例，故把本卡对角线概率置零。
        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)
    
    # 每句话按文找图概率抽一张容易混淆的错误图片。
    images_neg = []    
    for b in range(bsz):
        # multinomial 按概率抽候选编号，而不是直接取最相似的一张。
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        images_neg.append(all_image[neg_idx])
    images_neg = torch.stack(images_neg, dim=0)   

    # select a negative text for each image
    # 每张图片同样按图找文概率抽一条错误文字。
    text_ids_neg = []
    text_masks_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(all_text_ids[neg_idx])
        text_masks_neg.append(all_text_masks[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)     
    text_masks_neg = torch.stack(text_masks_neg, dim=0)      

    # text_labels is not used in ITM loss
    # 错图＋原文：图片列表仍按 DataModule 的结构包一层。
    batch_imgs_neg = {"image":[images_neg], "text_ids":batch["text_ids"], "text_labels":batch["text_labels"], "text_masks":batch["text_masks"]}
    infer_imags_neg = pl_module.infer(batch_imgs_neg, mask_text=False, mask_image=False)
    
    # 原图＋错文：组成另一种错误配对。
    batch_text_neg = {"image":batch["image"], "text_ids":text_ids_neg, "text_labels":batch["text_labels"], "text_masks":text_masks_neg}
    infer_text_neg = pl_module.infer(batch_text_neg, mask_text=False, mask_image=False)

    # 三组整体图文向量按标签顺序拼成 [3B,D]。
    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_imags_neg["cls_feats"], infer_text_neg["cls_feats"]], dim=0)

    # ITM 分类头输出 [3B,2]，分别是错误与正确配对的原始分数。
    itm_logits = pl_module.itm_score(all_cls_feats)
    # 交叉熵推动正例选中 1，两组负例选中 0。
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


# The implementation of image-text contrastive refers to open_clip (https://github.com/mlfoundations/open_clip)
def compute_itc(pl_module, batch, aggregate=True):
    """ITC：图文先分开编码，两套 CLS 表示各算一次双向对比损失。"""
    # pl_module.logit_scale.data = torch.clamp(pl_module.logit_scale.data, 0, 4.6052)

    # ITC 先分别编码图像与文字，尚未使用联合图文序列。
    infer_imag = pl_module.infer_image(batch, mask_image=False)
    infer_text = pl_module.infer_text(batch, mask_text=False)

    # 单模态专家路径得到 [B,D] 的整图与整句单位向量。
    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    # exp 保证相似度缩放为正；数值尺度影响后续 softmax 的尖锐程度。
    logit_scale = pl_module.logit_scale.exp().mean()

    # 重走 VL-FFN 后段得到第二套可对比向量。
    image_vlffn_features = infer_imag["cls_vlffn_feats"]
    text_vlffn_features = infer_text["cls_vlffn_feats"]
    logit_vl_scale = pl_module.logit_vl_scale.exp().mean()

    # 多卡训练时汇总各卡向量，让其他 GPU 的样本也成为负例。
    if aggregate:
        # 原版假定分布式进程组已初始化，即使只用一张卡也走此路径。
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        # 每个 GPU 都建立 world_size 个同形状空张量接收各卡特征。
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        # all_gather 后每个进程都拥有其他卡的图文向量。
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        # 本卡张量直接保留在图中，其他卡的特征由 all_gather 提供。
        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        # [总图片数,D] @ [D,总文本数] -> 图文两两相似度矩阵。
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        # 第二套 VL-FFN 向量也重复相同的多卡汇总过程。
        gathered_image_vlffn_features = [
            torch.zeros_like(image_vlffn_features) for _ in range(world_size)
        ]
        gathered_text_vlffn_features = [
            torch.zeros_like(text_vlffn_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_vlffn_features, image_vlffn_features)
        dist.all_gather(gathered_text_vlffn_features, text_vlffn_features)

        all_image_vlffn_features = torch.cat(
            [image_vlffn_features]
            + gathered_image_vlffn_features[:rank]
            + gathered_image_vlffn_features[rank + 1 :]
        )
        all_text_vlffn_features = torch.cat(
            [text_vlffn_features]
            + gathered_text_vlffn_features[:rank]
            + gathered_text_vlffn_features[rank + 1 :]
        )
        # this is needed to send gradients back everywhere.
        # 第二组表示来自 VL-FFN 路径，也做一套图文对比。
        logits_per_vlffn_image = logit_vl_scale * all_image_vlffn_features @ all_text_vlffn_features.t()
        logits_per_vlffn_text = logits_per_vlffn_image.t()

    else:
        # 本卡向量直接形成相似度矩阵；原版此分支没有赋值 VL-FFN logits。
        # 因为后面仍会读取 logits_per_vlffn_image，不能直接设 aggregate=False。
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    # 第 i 张图和第 i 句话属于正确配对，所以分类目标是对角线 0..N-1。
    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    # 图找文和文找图各算一次交叉熵，再取平均。
    itc_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth)
        + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    # 第二套 VL-FFN 图文向量也计算对称交叉熵。
    itc_vlffn_loss = (
        F.cross_entropy(logits_per_vlffn_image.float(), ground_truth)
        + F.cross_entropy(logits_per_vlffn_text.float(), ground_truth)
    ) / 2

    # 两套目标再平均，得到本函数最终参与反向传播的 ITC 损失。
    itc_total_loss = (itc_loss + itc_vlffn_loss) * 0.5

    # 字典中的 logits 之后还要给 ITM 选难负例使用。
    ret = {
        "itc_loss": itc_total_loss,
        "itc_i2t_logits": logits_per_image,
        "itc_t2i_logits": logits_per_text,
        "itc_labels": ground_truth,
        "itc_logit_scale": logit_scale,
        "itc_logit_vl_scale": logit_vl_scale,
    }

    # train/val 采用不同的状态统计器，避免指标混在一起。
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
    scale = getattr(pl_module, f"{phase}_itc_logit_scale")(ret["itc_logit_scale"])
    # i2t、t2i 分别统计图找文和文找图的 batch 内正确率。
    i2t_acc = getattr(pl_module, f"{phase}_itc_i2t_accuracy")(
        ret["itc_i2t_logits"], ret["itc_labels"]
    )
    t2i_acc = getattr(pl_module, f"{phase}_itc_t2i_accuracy")(
        ret["itc_t2i_logits"], ret["itc_labels"]
    )
    pl_module.log(f"itc/{phase}/loss", loss)
    pl_module.log(f"itc/{phase}/logit_scale", scale)
    pl_module.log(f"itc/{phase}/i2t_accuracy", i2t_acc)
    pl_module.log(f"itc/{phase}/t2i_accuracy", t2i_acc)

    # 下面记录第二套 VL-FFN 分支的温度和双向准确率。
    vl_scale = getattr(pl_module, f"{phase}_itc_vl_logit_scale")(ret["itc_logit_vl_scale"])
    vl_i2t_acc = getattr(pl_module, f"{phase}_itc_vl_i2t_accuracy")(
        logits_per_vlffn_image, ret["itc_labels"]
    )
    vl_t2i_acc = getattr(pl_module, f"{phase}_itc_vl_t2i_accuracy")(
        logits_per_vlffn_text, ret["itc_labels"]
    )
    pl_module.log(f"itc/{phase}/vl_logit_scale", vl_scale)
    pl_module.log(f"itc/{phase}/vl_i2t_accuracy", vl_i2t_acc)
    pl_module.log(f"itc/{phase}/vl_t2i_accuracy", vl_t2i_acc)

    return ret


def compute_irtr(pl_module, batch, aggregate=True):
    """下游图文检索微调：只使用单模态图像/文字向量。"""
    # pl_module.logit_scale.data = torch.clamp(pl_module.logit_scale.data, 0, 4.6052)

    # _ft 版本不计算第二套 VL-FFN 表示，比预训练 ITC 路径更简洁。
    infer_imag = pl_module.infer_image_ft(batch, mask_image=False)
    infer_text = pl_module.infer_text_ft(batch, mask_text=False)

    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()

    # 多卡把每卡的正例都纳入同一相似度矩阵，增加负例数量。
    if aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    # 图 i 对应文 i，正确位置位于相似度矩阵对角线。
    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    # 与 CLIP 一样对行和列各算交叉熵，平均成检索微调损失。
    irtr_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth)
        + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    ret = {
        "irtr_loss": irtr_loss,
        "irtr_i2t_logits": logits_per_image,
        "irtr_t2i_logits": logits_per_text,
        "irtr_labels": ground_truth,
        "irtr_logit_scale": logit_scale,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    scale = getattr(pl_module, f"{phase}_irtr_logit_scale")(ret["irtr_logit_scale"])
    i2t_acc = getattr(pl_module, f"{phase}_irtr_i2t_accuracy")(
        ret["irtr_i2t_logits"], ret["irtr_labels"]
    )
    t2i_acc = getattr(pl_module, f"{phase}_irtr_t2i_accuracy")(
        ret["irtr_t2i_logits"], ret["irtr_labels"]
    )
    pl_module.log(f"irtr/{phase}/loss", loss)
    pl_module.log(f"irtr/{phase}/logit_scale", scale)
    pl_module.log(f"irtr/{phase}/i2t_accuracy", i2t_acc)
    pl_module.log(f"irtr/{phase}/t2i_accuracy", t2i_acc)

    return ret


def compute_vqa(pl_module, batch):
    """视觉问答：融合图片与问题，预测候选答案的软分数。"""
    # 推理路径先把问题文字与图片 token 放入同一 Transformer。
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    # 分类头把 [B,D] 整体向量投影成 [B,答案候选数]。
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    # VQAv2 允许多个标注员给不同答案，因此目标是软分数矩阵而非单个 ID。
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    # 每个问题对应若干答案编号和各自的人工一致度分数。
    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    # 把稀疏答案编号及分数填入密集的目标矩阵。
    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    # BCEWithLogits 同时处理所有答案项；输入是原始 logits，不先 sigmoid。
    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    """NLVR2：同一句话分别配两张图，再判断整体陈述真假。"""
    # 第一张图用类型 1，第二张图用类型 2，避免模型混淆角色。
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    # 两个 [B,D] 图文向量拼成 [B,2D]。
    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    # 分类头输出 [B,2] 的真假分数。
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    # answers 是 0/1 标签，转成与模型相同设备的 long 张量。
    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    # 二分类交叉熵训练真假判断。
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    # 训练阶段整体统计；验证阶段还要拆分 dev/test 两部分。
    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        # batch 里混合了 dev/test 表，按来源名拆开分别报告。
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module, split="test"):
    """检索评估：先独立编码全部图片/文字，再统计双向 R@1/5/10。"""
    # no_grad 表示评估不反传；分布式信息用于和训练环境保持一致。
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 验证和测试使用不同 Arrow 划分；都不抽随机错配样本。
    if split == "val":
        rank_zero_info("Use val set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    else:
        rank_zero_info("Use test set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()

    # 与训练阶段保持同一个 BERT tokenizer。
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    # 一次读 32 条文字；collate 负责补齐和创建 text_ids 等字段。
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    # 图片数据集设 image_only=True，使每张图只出现一次。
    if split == "val":
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
            image_only=True
        )
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(
            image_only=True
        )

    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    # 先把全部文字 batch 预取到内存/显存，避免反复读取 Arrow 文件。
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    # 每条文字对应的真实图片编号；一张图可以有多条描述。
    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    rank_zero_info("len(tiids): {}".format(len(tiids)))

    # 图片同样预取，并记录每张图片的编号。
    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(
            {
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
            }
        )
    iids = list()
    for pre in image_preload:
        iids += pre["img_index"]
    iids = torch.tensor(iids)

    rank_zero_info("len(iids): {}".format(len(iids)))

    # 独立编码所有句子，不需要为每一对候选图文重跑 Transformer。
    txt_cls_feats = list()
    for txt_batch in text_preload:
        # 混合精度推理降低显存占用。
        with torch.cuda.amp.autocast():
            cls_feats = pl_module.infer_text_ft(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    }
                )["cls_feats"]
        txt_cls_feats.append(cls_feats)

    # 独立编码所有图片。
    img_cls_feats = list()
    for img_batch in image_preload:
        with torch.cuda.amp.autocast():
            cls_feats = pl_module.infer_image_ft(
                    {
                        "image": img_batch["image"],
                    }
                )["cls_feats"]
        img_cls_feats.append(cls_feats)

    txt_cls_feats = torch.cat(txt_cls_feats)
    img_cls_feats = torch.cat(img_cls_feats)

    rank_zero_info("txt_cls_feats.size(): {}\t{}".format(txt_cls_feats.size(), split))
    rank_zero_info("img_cls_feats.size(): {}\t{}".format(img_cls_feats.size(), split))

    # [图片数,D]×[D,文字数] → 全部候选的相似度矩阵。
    scores = img_cls_feats @ txt_cls_feats.t()

    rank_zero_info("scores.size(): {}".format(scores.size(), split))

    # dim=1：每张图片按文字分数排序，分别检查前 10、5、1 名。
    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices.to(tiids.device)]
    topk5_iids = tiids[topk5.indices.to(tiids.device)]
    topk1_iids = tiids[topk1.indices.to(tiids.device)]

    # 只要前 K 条文字中存在与该图同编号的描述，就算检索命中。
    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    # dim=0：每句话沿图片方向排名，得到反方向的检索结果。
    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices.to(iids.device)]
    topk5_iids = iids[topk5.indices.to(iids.device)]
    topk1_iids = iids[topk1.indices.to(iids.device)]

    # 根据图片编号而非单纯对角线判断，兼容一张图对应多条 caption。
    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


@torch.no_grad()
def compute_irtr_recall_with_rerank(pl_module, split="test"):
    """两阶段检索：先用 ITC 向量筛前 K，再用融合 ITM 重新排序。"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if split == "val":
        rank_zero_info("Use val set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    else:
        rank_zero_info("Use test set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()

    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    if split == "val":
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
            image_only=True
        )
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(
            image_only=True
        )

    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    text_ids_list = list()
    text_masks_list = list()
    text_labels_list = list()
    for pre in text_preload:
        tiids += pre["img_index"]
        text_ids_list.append(pre["text_ids"])
        text_masks_list.append(pre["text_masks"])
        text_labels_list.append(pre["text_labels"])
    tiids = torch.tensor(tiids)
    # Rerank 阶段需要原始文字 ID 和 mask，不能只保存单个 CLS 向量。
    all_text_ids = torch.cat(text_ids_list)
    all_text_masks = torch.cat(text_masks_list)
    all_text_labels = torch.cat(text_labels_list)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(
            {
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
            }
        )
    iids = list()
    image_list = list()
    for pre in image_preload:
        iids += pre["img_index"]
        image_list.append(pre["image"][0])
    iids = torch.tensor(iids)
    # 同样保留每张图片张量供图文联合 Transformer 再次编码。
    all_image = torch.cat(image_list)

    txt_cls_feats = list()
    for txt_batch in text_preload:
        with torch.cuda.amp.autocast():
            cls_feats = pl_module.infer_text_ft(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    }
                )["cls_feats"]
        txt_cls_feats.append(cls_feats)

    img_cls_feats = list()
    for img_batch in image_preload:
        with torch.cuda.amp.autocast():
            cls_feats = pl_module.infer_image_ft(
                    {
                        "image": img_batch["image"],
                    }
                )["cls_feats"]
        img_cls_feats.append(cls_feats)

    txt_cls_feats = torch.cat(txt_cls_feats)
    img_cls_feats = torch.cat(img_cls_feats)

    # 第一阶段：便宜的双编码器点积，先筛掉大部分候选。
    scores = img_cls_feats @ txt_cls_feats.t()

    rank_zero_info("scores.size(): {}".format(scores.size(), split))


    # 未进入前 K 的配对先填很低的分数，避免最终排到前面。
    scores_i2t = torch.full((len(iids), len(tiids)), -100.0).to(pl_module.device)
    
    # k_test 默认 32：每个查询只让 32 个最相似候选走昂贵的 ITM 融合。
    k_test = pl_module.hparams.config["k_test"]
    num_tasks = world_size
    step = scores.size(0) // num_tasks + 1
    # 各 GPU 分担不同图片行，避免重复 rerank 同一批查询。
    start = rank * step
    end = min(scores.size(0), start+step)

    for i, sims in enumerate(scores[start:end]): 
        if i%100 == 0:
            rank_zero_info("TR Rerank: {}".format(i))
        # 当前图片在所有文字里取第一阶段分数最高的 K 条。
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        cur_images = all_image[start+i].repeat(k_test, 1, 1, 1)
        cur_text_ids = all_text_ids[topk_idx]
        cur_text_masks = all_text_masks[topk_idx]
        cur_text_labels = all_text_labels[topk_idx]
        
        cur_rerank_batch = {"image":[cur_images], "text_ids":cur_text_ids, "text_labels":cur_text_labels, "text_masks":cur_text_masks}
        # 第二阶段让图像 patch 和文字 token 一起进入融合 Transformer。
        infer_rerank = pl_module.infer(cur_rerank_batch, mask_text=False, mask_image=False)
        itm_logits = pl_module.itm_score(infer_rerank["cls_feats"])
        # ITM 分类头第 1 列是“匹配”类别的原始分数。
        itm_scores = itm_logits[:,1]
        scores_i2t[start+i,topk_idx] = itm_scores

    # 反方向把文字当查询，故将第一阶段矩阵转置。
    scores = scores.t()
    scores_t2i = torch.full((len(tiids), len(iids)), -100.0).to(pl_module.device)
    
    step = scores.size(0) // num_tasks + 1
    start = rank * step
    end = min(scores.size(0), start+step)    
    
    for i,sims in enumerate(scores[start:end]): 
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        cur_images = all_image[topk_idx]
        cur_text_ids = all_text_ids[start+i].repeat(k_test, 1) 
        cur_text_masks = all_text_masks[start+i].repeat(k_test, 1) 
        cur_text_labels = all_text_labels[start+i].repeat(k_test, 1)
        cur_rerank_batch = {"image":[cur_images], "text_ids":cur_text_ids, "text_labels":cur_text_labels, "text_masks":cur_text_masks}
        infer_rerank = pl_module.infer(cur_rerank_batch, mask_text=False, mask_image=False)
        itm_logits = pl_module.itm_score(infer_rerank["cls_feats"])
        itm_scores = itm_logits[:,1]
        scores_t2i[start+i, topk_idx] = itm_scores

    # 等所有 GPU 都完成各自负责的行，再合并两张 rerank 分数表。
    dist.barrier()   
    torch.distributed.all_reduce(scores_i2t, op=torch.distributed.ReduceOp.SUM) 
    torch.distributed.all_reduce(scores_t2i, op=torch.distributed.ReduceOp.SUM)    


    # 最终分数是 ITM 二阶段分数加上原来的 ITC 相似度。
    scores_t2i = scores_t2i + scores
    scores_i2t = scores_i2t + scores.t()

    topk10 = scores_i2t.topk(10, dim=1)
    topk5 = scores_i2t.topk(5, dim=1)
    topk1 = scores_i2t.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores_t2i.topk(10, dim=1)
    topk5 = scores_t2i.topk(5, dim=1)
    topk1 = scores_t2i.topk(1, dim=1)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    ir_r5 = (tiids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    ir_r1 = (tiids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def vqa_test_wrapup(outs, model_name, output_dir):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        # os.makedirs("result", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")
