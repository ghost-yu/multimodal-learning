# functools.partial 将 mlm_collator 预先绑定到数据集的 collate 函数。
import functools

# LightningDataModule 是训练器调用 prepare_data/setup/dataloader 的统一接口。
from pytorch_lightning import LightningDataModule
# DataLoader 逐批读取，ConcatDataset 拼接 COCO/VG/SBU/GCC 等来源。
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
# DistributedSampler 将数据按 GPU 进程切分，避免每卡重复读相同样本。
from torch.utils.data.distributed import DistributedSampler

# 注册表：数据集名字映射到对应 DataModule 类。
from . import _datamodules


class MTDataModule(LightningDataModule):
    """Multi-Task DataModule：统一管理多种数据集和三种划分。"""
    def __init__(self, _config, dist=False):
        # 例如 ["coco","vg","sbu","gcc"]。
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        # 每个名字建立一套负责该数据来源的 DataModule。
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        # 这些来源共用第一个数据模块的 tokenizer、batch 和 worker 配置。
        self.batch_size = self.dms[0].batch_size
        self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist

    def prepare_data(self):
        # 各来源分别检查/准备底层文件。
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        # 每套 DataModule 先建立各自训练/验证/测试 Dataset。
        for dm in self.dms:
            dm.setup(stage)

        # 训练集纵向拼接，DataLoader 看到的是统一样本序列。
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        self.tokenizer = self.dms[0].tokenizer

        # collate 把图片/文字列表堆成张量，同时执行 MLM 遮词。
        self.collate = functools.partial(
            self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator,
        )

        # 原版 run.py 传入 dist=True，因此为每个划分建立分布式采样器。
        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        # 每张 GPU 的 batch_size 来自 per_gpu_batchsize，不是配置中的总 batch_size。
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # sampler 负责分卡与训练集打乱。
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            # 默认 DataLoader 拼接器不知道如何处理 Arrow/MLM，必须用自定义版。
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        # 验证可临时覆盖单卡 batch 大小。
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self):
        # 测试 sampler 不打乱，方便把预测对应回原始样本。
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
