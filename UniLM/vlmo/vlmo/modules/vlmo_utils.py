# torch 提供优化器、张量等训练基础组件。
import torch
# 以下两个标准库由原项目引入；本文件当前没有直接使用它们。
import random
import json

# AdamW 是带权重衰减的 Adam；学习率调度器负责训练中逐步改变学习率。
from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
# all_gather 用于跨 GPU 汇集张量；此文件保留原项目的导入。
from vlmo.modules.dist_utils import all_gather
# Recall 用于图文检索评估；带 rerank 的版本在本文件未直接调用。
from vlmo.modules.objectives import compute_irtr_recall, compute_irtr_recall_with_rerank
# Accuracy 统计分类准确率，VQAScore 统计 VQA 分数，Scalar 统计标量均值。
from vlmo.gadgets.my_metrics import Accuracy, VQAScore, Scalar
# 只让分布式训练的主进程打印日志，避免多个 GPU 重复输出。
from pytorch_lightning.utilities.distributed import rank_zero_info


def set_metrics(pl_module):
    """为启用的任务创建训练、验证指标容器，并挂到模型对象上。"""
    # 分别为训练和验证准备指标；f"{split}_..." 会生成 train_... 或 val_... 名称。
    for split in ["train", "val"]:
        # loss_names 是配置字典，例如 {"mlm": 1, "itm": 1, "vqa": 0}。
        for k, v in pl_module.hparams.config["loss_names"].items():
            # 权重小于 1 表示这个任务暂时不启用，因此不创建它的指标。
            if v < 1:
                continue
            if k == "vqa":
                # setattr(对象, 名字, 值) 等价于按动态名字给对象添加属性。
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                # NLVR2 的验证数据还区分 dev 与 test，因而需要额外的指标容器。
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                # 图文检索分别记录“以图找文”和“以文找图”的准确率。
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())
            elif k == "itm":
                # ITM 是图文是否匹配的二分类任务。
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itc":
                # ITC 除基础对比学习外，还记录 VL 融合分支的两个方向指标。
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())

                setattr(pl_module, f"{split}_{k}_vl_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_vl_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_vl_logit_scale", Scalar())
            else:
                # MLM 等其余任务按普通 accuracy 与 loss 记录。
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    """每个 epoch 结束时读取指标、写日志、清零，并生成汇总分数。"""
    # pl_module.training 是 PyTorch 模型的布尔状态：True 表示训练模式。
    phase = "train" if pl_module.training else "val"
    # 汇总指标用于日志和 checkpoint 选择；它不是反向传播的 loss。
    the_metric = 0

    # 检索 Recall 计算成本较高，只在配置开启且当前处于验证阶段时执行。
    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        # 返回六个 Recall@K：图搜文 IR 与文搜图 TR，各有 @1/@5/@10。
        (val_ir_r1, val_ir_r5, val_ir_r10, val_tr_r1, val_tr_r5, val_tr_r10) = compute_irtr_recall(pl_module, split="val")
        # .item() 将单元素张量变成 Python 数字，便于求均值和打印。
        val_avg = (val_ir_r1.item() + val_ir_r5.item() + val_ir_r10.item() + val_tr_r1.item() + val_tr_r5.item() + val_tr_r10.item()) / 6.0
        # logger.experiment 是底层 TensorBoard 写入器；global_step 是当前优化步数。
        pl_module.logger.experiment.add_scalar(
            "recalls/val_avg", val_avg, pl_module.global_step
        )

        # 同样计算测试集 Recall；原项目会在每次验证时执行这一步。
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module, split="test")
        test_avg = (ir_r1.item() + ir_r5.item() + ir_r10.item() + tr_r1.item() + tr_r5.item() + tr_r10.item()) / 6.0
        # 这些日志项可在 TensorBoard 中看到，每项保留单独的检索方向和 K 值。
        pl_module.logger.experiment.add_scalar(
            "recalls/test_avg", test_avg, pl_module.global_step
        )

        print("val_avg:{}, test_avg:{}".format(val_avg, test_avg))
        print("test ir_r1:{}, ir_r5:{}, ir_r10:{}, tr_r1:{}, tr_r5:{}, tr_r10:{}".format(ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10))
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        # 把验证集均值计入本轮的总评分。
        the_metric += val_avg


    # 逐一处理每个已启用任务。
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        # 每个分支会把自己的“主要评估值”写入 value。
        value = 0

        if loss_name == "vqa":
            # getattr(对象, 名字) 是按动态名字读属性，和上面的 setattr 相配。
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            # compute() 汇总整个 epoch 的累计值；reset() 清零以便下个 epoch 使用。
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "nlvr2":
            # 训练阶段只有 train；验证阶段分别记录 dev 与 test。
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value_dev = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value_dev)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value_test = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value_test)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
                value = value_dev
        elif loss_name == "irtr":
            # 检索任务将两个方向的准确率相加，作为该任务的汇总值。
            value_i2t = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()
            
            value_t2i = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()

            value = value_i2t + value_t2i
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            # 二分类任务使用准确率作为主要指标，同时记录平均 loss。
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itc":
            # ITC 记录基础图搜文/文搜图，以及 VL 分支的两个准确率。
            value_i2t = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()
            
            value_t2i = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            value_vl_i2t = getattr(pl_module, f"{phase}_{loss_name}_vl_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/vl_i2t_accuracy_epoch", value_vl_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_vl_i2t_accuracy").reset()
            
            value_vl_t2i = getattr(pl_module, f"{phase}_{loss_name}_vl_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/vl_t2i_accuracy_epoch", value_vl_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_vl_t2i_accuracy").reset()

            # 当前实现只把基础两个方向计入汇总分数，VL 指标仅做日志展示。
            value = value_i2t + value_t2i
        else:
            # 例如 MLM：读准确率、写日志，再把容器清零。
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        # 把该任务的主要指标累加到统一分数中。
        the_metric += value

    # run.py 的 checkpoint 监控项就可以读取 val/the_metric。
    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    """检查 token type embedding 的梯度是否缺失或总和为零。"""
    # grad 为 None 通常表示该参数没有参与当前反向传播。
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        # .sum() 把所有梯度元素相加；注意正负抵消也可能让和为零。
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    """把本轮要执行的任务名放进 current_tasks，供模型 forward 选择分支。"""
    # 列表推导式依次取出 loss_names 的键 k 和权重 v，仅保留启用项。
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    """建立优化器及学习率调度器；由 Lightning 的 configure_optimizers 调用。"""
    # learning_rate 是基础学习率；weight_decay 是权重衰减系数。
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    # 名字含下列字符串的参数（偏置或归一化层）不做权重衰减。
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    # 下游任务的分类头可以使用与主干不同的学习率。
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    # 分类头实际学习率 = 基础学习率 lr × lr_mult。
    lr_mult = pl_module.hparams.config["lr_mult"]
    # end_lr 和 decay_power 控制多项式学习率衰减曲线。
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    # optim_type 选择 adamw / adam / sgd。
    optim_type = pl_module.hparams.config["optim_type"]

    # 原项目保留了 names 列表，但后续没有使用；真正分组在下面完成。
    names = [n for n, p in pl_module.named_parameters()]
    # 按“是否衰减”×“是否任务头”分成四组，避免一刀切地设置训练超参数。
    optimizer_grouped_parameters = [
        {
            # 组 1：主干中的普通权重；需要衰减，学习率为 lr。
            "params": [
                p
                # named_parameters() 产出 (参数名, 参数张量)。
                for n, p in pl_module.named_parameters()
                # any(...) 判断名字是否包含列表中任一字符串。
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            # 组 2：主干的 bias / norm；不做权重衰减。
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            # 组 3：下游任务头普通权重；学习率乘 lr_mult。
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            # 组 4：任务头的 bias / norm；既不衰减，也使用放大后的学习率。
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    # 使用四组参数创建具体优化器；AdamW 是默认实验中常用的选择。
    if optim_type == "adamw":
        optimizer = AdamW(
            # eps 防止除零；betas 控制梯度一阶/二阶滑动平均。
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    # 若 Trainer 没有明确给 max_steps，就根据数据批次数、epoch 数和梯度累积推算。
    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps==-1:
        max_steps = (
            # len(dataloader) 是每个 epoch 的 batch 数。
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            # 多个 batch 累积一次梯度，优化器更新次数相应减少。
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        # 若明确配置了 max_steps，则直接使用该值。
        max_steps = pl_module.trainer.max_steps

    # warmup 可以是整数步数，也可以是训练总步数的浮点比例。
    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)
    # 预热阶段通常从很小的学习率逐步升到设定学习率。
    rank_zero_info("Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

    # 预热之后二选一：余弦曲线衰减，或多项式曲线衰减。
    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    # interval="step" 表示每一次优化器更新后调整学习率，而非每个 epoch。
    sched = {"scheduler": scheduler, "interval": "step"}

    # Lightning 要求的返回格式：优化器列表与调度器配置列表。
    return (
        [optimizer],
        [sched],
    )
