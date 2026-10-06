# os 读取多卡运行时的环境变量，也负责创建日志目录。
import os
# 深拷贝配置，避免后续修改波及 Sacred 保存的原始配置。
import copy
# Lightning 封装训练循环、验证、日志和多 GPU 调度。
import pytorch_lightning as pl

# Sacred 的 ex 保存默认配置和所有 named_config。
from vlmo.config import ex
# VLMo 是真正的模型；MTDataModule 把多份数据集整理成 batch。
from vlmo.modules import VLMo
from vlmo.datamodules.multitask_datamodule import MTDataModule

# Lightning 的集群环境接口；OMPI 用于集群启动的多进程训练。
from pytorch_lightning.plugins import environments as pl_env
# 只在全局 rank 0 打印，避免每张 GPU 重复输出。
from pytorch_lightning.utilities.distributed import rank_zero_info


class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
    def __init__(self):
        # 初始化 Lightning 的集群环境父类。
        super().__init__()

    # def creates_children(self) -> bool:
    #     # return True if the cluster is managed (you don't launch processes yourself)
    #     assert (
    #         "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
    #     )  # this cluster is managed
    #     return True

    @property
    def creates_processes_externally(self):
        # True 表示 MPI 等外部启动器已经创建好多个训练进程。
        return True

    def world_size(self) -> int:
        # world_size 是参与训练的进程总数，通常等于 GPU 总数。
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def set_world_size(self, size: int):
        # 数值来自外部环境变量，所以这里无需在 Python 内修改。
        pass

    def global_rank(self) -> int:
        # global_rank 是整个集群里的进程编号，从 0 开始。
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def set_global_rank(self, rank: int):
        # 编号由 MPI 启动器提供，不在此处重新设置。
        pass

    def local_rank(self) -> int:
        # local_rank 是当前机器上的进程编号，决定绑定哪张 GPU。
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        # 多台机器训练时，NODE_RANK 指当前机器的编号。
        if "NODE_RANK" in os.environ:
            # 环境变量是字符串，需要转换为整数。
            return int(os.environ["NODE_RANK"])
        else:
            # 单机运行时只有第 0 台机器。
            return 0

    def master_address(self) -> str:
        # 各进程通过主节点地址建立通信。
        return os.environ["MASTER_ADDR"]

    def master_port(self) -> int:
        # 通信端口也由外部运行环境指定。
        return int(os.environ["MASTER_PORT"])


def get_cluster_plugin(num_gpus=1, num_nodes=1):
    # 跨节点，或单节点但已处于 MPI 环境时，使用上面的适配器。
    if num_nodes > 1 or (
        num_nodes == 1 and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        # 只让主进程打印所选集群方案。
        rank_zero_info("ClusterPlugin: using OMPI Cluster Environment")
        # 返回环境对象给 Lightning Trainer。
        return OMPIClusterEnvironment()
    # 有 GPU、又没有 MPI 时，使用 Lightning 自带的本地环境。
    if num_gpus >= 1:
        rank_zero_info("ClusterPlugin: using Lightning Cluster Environment")
        return pl_env.LightningEnvironment()
    # 无 GPU 时不额外配置集群插件。
    return None


@ex.automain
def main(_config):
    # Sacred 会把默认配置、named_config 和命令行覆盖合并成 _config。
    _config = copy.deepcopy(_config)
    # 固定 Python/NumPy/PyTorch 随机数，尽可能复现实验。
    pl.seed_everything(_config["seed"])

    # DataModule 负责准备、拼接各数据集，以及创建训练/验证 DataLoader。
    dm = MTDataModule(_config, dist=True)

    # 根据配置创建共享注意力＋多路 FFN 的 VLMo 网络。
    model = VLMo(_config)
    # 实验名将出现在日志目录名中。
    exp_name = f'{_config["exp_name"]}'

    # 如果日志目录不存在，先建立它。
    os.makedirs(_config["log_dir"], exist_ok=True)
    # Checkpoint 回调负责定期保存模型参数和最后一次状态。
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # -1 表示保留所有满足保存条件的检查点。
        save_top_k=-1,
        # 打印检查点保存位置。
        verbose=True,
        # 用验证集综合指标作为监控量。
        monitor="val/the_metric",
        # 指标越大越好。
        mode="max",
        # 额外保存 last.ckpt，方便意外中断后恢复。
        save_last=True,
    )
    # TensorBoardLogger 把损失和学习率等曲线写入磁盘。
    logger = pl.loggers.TensorBoardLogger(
        # 日志根目录。
        _config["log_dir"],
        # 文件夹名含实验名、种子和初始权重文件名。
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    # 每个优化步骤记录学习率，便于检查 warmup/衰减是否正确。
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # 把要启用的两个回调交给 Trainer。
    callbacks = [checkpoint_callback, lr_callback]

    # num_gpus 可以是整数，也可以是 GPU 编号列表。
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    # 梯度累积：总目标 batch / 单卡 batch / GPU 数 / 节点数。
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    # 输出实际累积步数，避免以为 per_gpu_batchsize 就是总 batch。
    rank_zero_info("grad_steps: {}".format(grad_steps))

    # max_steps 不为空时按优化步数停止，否则由 max_epoch 控制。
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # 默认从头启动；下面可查找旧的 last.ckpt 继续训练。
    resume_ckpt = None
    if _config["resume_during_training"]:
        # 依次查找 TensorBoardLogger 可能创建的 version_0、version_1 等目录。
        for index in range(100):
            ckpt_path = os.path.join(_config["log_dir"], f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}', "version_{}/checkpoints/last.ckpt".format(index))
            # 找到时记录路径；如果存在多个版本，最终选择编号最大的那个。
            if os.path.exists(ckpt_path):
                resume_ckpt = ckpt_path
    
    # 打印最终选中的恢复文件；None 表示不恢复。
    rank_zero_info("resume_ckpt: {}".format(resume_ckpt))

    # 根据 GPU 和节点数选择上文定义的集群环境。
    cluster_plugin = get_cluster_plugin(
        _config["num_gpus"], _config["num_nodes"]
    )
    # Lightning 的 plugins 参数接收列表。
    plugin_list = [cluster_plugin]
    rank_zero_info("plugin_list: {}".format(plugin_list))

    # 分片训练把优化状态和参数分散到多卡，降低单卡显存。
    if _config["use_sharded_training"]:
        rank_zero_info("Using ddp sharded")
        distributed_strategy = "ddp_sharded"
    else:
        # 常规 DDP：每卡各有一份模型，梯度在卡间同步。
        distributed_strategy = "ddp"

    # Trainer 才是实际执行 fit/test 循环的对象。
    trainer = pl.Trainer(
        # 每台机器使用的 GPU 数量或 GPU 编号。
        gpus=_config["num_gpus"],
        # 多机训练的机器数量。
        num_nodes=_config["num_nodes"],
        # 16 通常表示半精度训练，节省显存。
        precision=_config["precision"],
        # 这个原版入口按 GPU 训练编写。
        accelerator="gpu",
        # 使用上面选出的 DDP 或分片策略。
        strategy=distributed_strategy,
        # cuDNN 自动尝试较快的卷积实现。
        benchmark=True,
        # 尽量使用确定性算法以便复现。
        deterministic=True,
        # 按步数训练时给一个足够大的 epoch 上限，真正停止由 max_steps 控制。
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        # 回调和日志器负责保存训练状态与曲线。
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        # 已由 DataModule 设置 sampler，这里不让 Lightning 再自动替换。
        replace_sampler_ddp=False,
        # 把前面算出的多次小 batch 梯度累积成一个大 batch 更新。
        accumulate_grad_batches=grad_steps,
        # 每 10 步记录一次日志。
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        # 存在恢复文件时从该训练状态继续。
        resume_from_checkpoint=resume_ckpt,
        # 打印模型各层和参数统计信息。
        weights_summary="top",
        # 调试开关会只跑少量 batch。
        fast_dev_run=_config["fast_dev_run"],
        # 验证间隔来自配置。
        val_check_interval=_config["val_check_interval"],
        # 使用前面选定的集群环境插件。
        plugins=plugin_list,
    )

    # 文字单模态预训练阶段仅更新文字相关参数，其余参数冻结。
    if _config["loss_names"]["textmlm"] > 0:
        # 先把所有参数的 requires_grad 设为 False。
        for param in model.parameters():
            param.requires_grad = False

        # 再按名字打开文字嵌入、文字 FFN、MLM 头等参数。
        for name, param in model.named_parameters():
            for key in ["text_embeddings", "token_type_embeddings", "mlp_text", "norm2_text", "mlm_score", "relative_position_bias_table", "transformer.norm"]:
                if key in name:
                    param.requires_grad = True

        # 列出每个参数是否训练，方便核对冻结规则。
        for name, param in model.named_parameters():
            rank_zero_info("{}\t{}".format(name, param.requires_grad))

    # test_only=False 执行训练；DataModule 会把数据交给模型的 training_step。
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        # 只测试时调用模型的 test_step，不更新权重。
        trainer.test(model, datamodule=dm)
