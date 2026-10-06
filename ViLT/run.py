# os.makedirs 建训练日志目录；不会创建模型或数据集。
import os
# copy.deepcopy 复制嵌套配置，避免后续对 _config 的修改影响原对象。
import copy
# PyTorch Lightning 的 Trainer 管训练/验证循环，回调负责存权重和日志。
import pytorch_lightning as pl

# ex 是 Sacred Experiment；@ex.automain 会将配置及命令行覆盖项传给 main。
from vilt.config import ex
# 主模型类：输入由 DataModule 产生的 batch，输出目标函数的损失或特征。
from vilt.modules import ViLTransformerSS
# MT = multitask，多任务 DataModule 负责选数据集、建 dataloader。
from vilt.datamodules.multitask_datamodule import MTDataModule


@ex.automain
def main(_config):
    # _config 是 Sacred 传来的配置字典；前导 _ 是框架约定的参数名。
    _config = copy.deepcopy(_config)
    # 为 Python/NumPy/PyTorch 等随机源设种子，帮助同配置实验重复。
    pl.seed_everything(_config["seed"])

    # DataModule 准备数据集与 dataloader；模型只关心收到的 batch。
    dm = MTDataModule(_config, dist=True)

    # 构造模型，此时只创建层/加载所设检查点；正式 batch 训练从 trainer.fit 开始。
    model = ViLTransformerSS(_config)
    # exp_name = experiment name，日志名称中的实验标识。
    exp_name = f'{_config["exp_name"]}'

    # exist_ok=True 允许目录已经存在，不会因此抛 FileExistsError。
    os.makedirs(_config["log_dir"], exist_ok=True)
    # 监视 val/the_metric；mode=max 表示越大越好，只保留最佳和最后权重。
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    # TensorBoardLogger 把随训练步变化的标量写入 log_dir，供可视化查看。
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    # lr = learning rate；每步记录学习率，便于检查 warmup/衰减是否执行。
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # num_gpus 可配置为设备个数整数，或设备编号列表；统一换成实际张数。
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    # 大的目标 batch 可由多个小 batch 累积梯度得到。
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    # max_steps 控制最多优化步数；None 时按 max_epoch 控制训练时长。
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # Trainer 不含模型结构；它读取 LightningModule 钩子执行训练/验证/测试。
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        # Lightning 每处理 grad_steps 个小 batch 才更新一次参数。
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        # fit 内部会调用模型的 training_step、反向传播和优化器更新。
        trainer.fit(model, datamodule=dm)
    else:
        # test_only=True 只测试已载入的权重，是否有权重由 load_path 配置决定。
        trainer.test(model, datamodule=dm)
