# random 用于抽错配图文和读取失败后的重试。
import random
# torch 将多个样本合成批量张量。
import torch
# io.BytesIO 把 Arrow 文件中的图片字节包装成可读取文件。
import io
# PyArrow 以内存映射方式读取预处理好的 *.arrow 数据。
import pyarrow as pa
# os.path 检查数据文件是否存在。
import os

# Pillow 负责解码 JPEG/PNG 图片。
from PIL import Image
# 配置名映射到图片缩放、增强等变换。
from vlmo.transforms import keys_to_transforms


class BaseDataset(torch.utils.data.Dataset):
    """把 Arrow 中的图像和多条描述展开为训练样本。"""
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=False,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        # 至少要有一种图像变换，例如训练增强或验证缩放。
        assert len(transform_keys) >= 1
        super().__init__()

        # 把字符串配置变成实际可调用的图片处理函数列表。
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        # names 指多个 Arrow 文件名，如 coco_train、vg_train。
        if len(names) != 0:
            # 只读取当前磁盘上实际存在的文件。
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            # 为每一行记录来自哪张 Arrow 表，供任务划分使用。
            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            # 多个表纵向拼成同一个可索引数据集。
            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                # 一张图可能对应多条 caption；转换为 Python 列表。
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    # 可选去掉同一张图的重复描述。
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        # index_mapper 将“样本序号”映射到“图片行号、caption 序号”。
        self.index_mapper = dict()

        if text_column_name != "" and not self.image_only:
            j = 0
            # 一张图片有 K 条描述，就展开为 K 个训练样本。
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        # 展平全部描述，供 tokenizer 或词频统计使用。
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        # 返回展开后的图文样本数，可能大于原始图片数。
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        # index_mapper 找到该样本对应的 Arrow 图片行。
        index, caption_index = self.index_mapper[index]
        # Arrow 中存的是图片字节，而不是普通 JPEG 文件路径。
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        # 解码图片并统一成 RGB 三通道。
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        # 先拿原始 PIL 图片。
        image = self.get_raw_image(index, image_key=image_key)
        # 每种图像变换产生一个视图，通常列表只含一个张量。
        image_tensor = [tr(image) for tr in self.transforms]
        # 保留图片编号与 caption 编号，检索评估要靠它们判对错。
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        # 随机抽另一张图，形成错配图文的候选；此处不保证一定不同图。
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        # 找到当前样本对应的第几张图、第几条描述。
        index, caption_index = self.index_mapper[raw_index]

        # 取人能读的原始句子，再交给 BERT tokenizer。
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            # 补齐至 max_text_len，让一个 batch 中所有文字长度一致。
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            # MLM collator 需要知道 CLS/SEP 等特殊 token 位置。
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        # 随机抽取一条其他位置的描述作为错配候选。
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        # 发生坏图片/坏文字时换一个随机样本重试。
        result = None
        while result is None:
            try:
                ret = dict()
                # 先写入图片及其编号。
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    # 同张图的非第 0 条 caption 标记为 replica。
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                # 依配置额外采样错误图片、错误描述。
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret


    def get_text_suite(self, index):
        # 纯文字预训练不用打开图片，只生成文字相关字段。
        result = None
        while result is None:
            try:
                ret = dict()
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret


    def collate(self, batch, mlm_collator):
        """把样本列表整理成图片 [B,C,H,W] 与文字 [B,T] 等张量。"""
        # B 是 DataLoader 本批样本数。
        batch_size = len(batch)
        # 不同任务样本的字段不完全相同，先汇总所有出现过的键。
        keys = set([key for b in batch for key in b.keys()])
        # 按字段聚合：每个键对应长度 B 的值列表。
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # 找出 image 和 false_image_0 等所有图片字段。
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]

        for img_key in img_keys:
            new_imgs = [tmp_img[0] for tmp_img in dict_batch[img_key]]
            # B 个 [C,H,W] 张量堆叠成 [B,C,H,W]。
            batch_new_imgs = torch.stack(new_imgs, dim=0)
            dict_batch[img_key] = [batch_new_imgs]

        # 找出 text、false_text_0 等所有文字字段。
        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            # 先把所有文字字段的编码摊平，统一交给 MLM 遮词器。
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                # 再把扁平输出切回当前文字字段对应的 B 行。
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                # 原始文字 ID 和有效位先以全零张量作为填充容器。
                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    # 将实际分词长度部分覆盖进当前样本对应行。
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                # 非 MLM 路径不需要词预测标签，全部设为忽略值 -100。
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
