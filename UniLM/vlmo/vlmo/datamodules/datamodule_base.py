# torch.distributed 用于让多卡只由 rank 0 首次下载 BERT tokenizer。
import torch

# DataModule 是 Lightning 训练入口的标准数据接口。
from pytorch_lightning import LightningDataModule
# DataLoader 根据 Dataset 按批提供数据。
from torch.utils.data import DataLoader
# BERT tokenizer 负责分词；两种 DataCollator 负责随机遮盖单词。
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)


def get_pretrained_tokenizer(from_pretrained):
    """多卡安全地加载 tokenizer，避免各进程同时下载。"""
    if torch.distributed.is_initialized():
        # rank 0 先完成缓存文件准备。
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        # 其他进程等 rank 0 完成后再继续。
        torch.distributed.barrier()
    # 现在所有进程都能从本地缓存读取相同的分词器。
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(LightningDataModule):
    """每种数据集共用的分词、遮词与 DataLoader 配置。"""
    def __init__(self, _config):
        super().__init__()

        # Arrow 数据文件所在目录。
        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        # 这里是每张 GPU 的 batch 大小，总 batch 还需乘 GPU 数和累积步数。
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        # 下列字段控制图片尺寸、文本长度和额外错配样本数。
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]
        self.text_only = _config["text_only"]

        # 若没有显式指定训练变换，使用 default_train。
        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        # 验证采用固定变换，避免随机增强干扰评估。
        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        # 原配置通常为 bert-base-uncased。
        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        # 整词遮盖时同一英文词的 WordPiece 子词会一起遮盖。
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        # mlm_prob 默认 0.15，即随机选约 15% 的词作为预测目标。
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        # 子类（如 COCO DataModule）必须提供真实 Dataset 类。
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        # 用子类指定的数据集类型和训练划分建立 Dataset。
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        # 验证数据使用固定图片变换。
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

        # 某些检索评估还需不含随机错配样本的纯验证 Dataset。
        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def make_no_false_test_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def setup(self, stage):
        # Lightning 在 fit/test 前调用 setup；标志防止重复创建。
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            # Dataset.get_text() 需要能访问同一个 BERT tokenizer。
            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        # shuffle=True 保证每轮读取顺序不同。
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # 锁页内存通常可加快 CPU 到 GPU 的数据传输。
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        # 验证不打乱，便于定位具体样本和复现实验。
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
