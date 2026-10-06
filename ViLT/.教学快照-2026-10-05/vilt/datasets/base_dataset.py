# random.randint 用来随机选负样本或坏样本读失败后的替代索引。
import random
# torch 张量拼 batch；Dataset 是 PyTorch 的单条样本接口。
import torch
# io.BytesIO 把 Arrow 中的图片字节包装成可供 PIL 读取的内存文件。
import io
# pyarrow 读取磁盘上的 .arrow 列式表，图片/文字数据按行存放。
import pyarrow as pa
# os.path.isfile 确认数据文件是否存在。
import os

# PIL.Image 负责从图片字节解码 RGB 图片；此时还不是 PyTorch 张量。
from PIL import Image
# keys_to_transforms 根据配置名创建 Resize/裁剪/增强等图片处理函数。
from vilt.transforms import keys_to_transforms


class BaseDataset(torch.utils.data.Dataset):
    # Dataset 的约定：__len__ 给长度，子类的 __getitem__ 用索引取单条样本。
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
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
        # 至少一种图片视图变换；其结果会放在 image 列表内。
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        # names 是不带 .arrow 后缀的数据文件名列表；按配置可组合多个分片。
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            # concat_tables 把多张 Arrow 表按行拼成统一 self.table。
            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                # 每图可能对应多条 caption；转成 Python 嵌套列表便于随机抽取。
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        # index_mapper: “数据集序号 -> (图片表行号, 该图的文字序号)”。
        self.index_mapper = dict()

        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        # @property 使外部能写 dataset.corpus；两层推导式展平所有图片的 captions。
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        # index_mapper 把数据集序号映射到表格行及该图的 caption 序号。
        index, caption_index = self.index_mapper[index]
        # 图片以字节存于 Arrow 表；先包成内存文件，再交给 PIL 解码。
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        # 可能有多个图像视图/增强版本，因此返回列表而不是单个 tensor。
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        # 随机抽图供匹配任务构造负例；此实现没有显式排除原图下标。
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        # raw_index 是 DataLoader 使用的数据集索引，先映射到表格图片行与 caption 序号。
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        # tokenizer 在 DataModule 中设置；padding 补到 max_text_len、truncation 截过长句。
        # return_special_tokens_mask 保留 CLS/SEP 等特殊位置，MLM collator 用它避开特殊词。
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        # rep 是第几个负文本的序号，决定返回键 false_text_0/false_text_1 等。
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
        # suite 是一套训练样本：真图文 + 按配置抽出的假图/假文。
        result = None
        while result is None:
            # 原实现读坏样本时随机换索引重试；若大量文件损坏可能反复重试。
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                # 按配置添加若干错误图像/文字，之后在 ITM 等目标中使用。
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        # Dataset 一次只给一条样本；collate 把多条样本拼成模型需要的 batch。
        batch_size = len(batch)
        # 收集本批所有单条样本出现过的键；set 去重。
        keys = set([key for b in batch for key in b.keys()])
        # dict_batch 的每个键先对应一个长度 B 的 Python 列表，再逐类转张量。
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # 既含 image 也含 false_image_0；两类都要对齐批内高宽。
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            # 每个 ii 是一次图片变换后的 [3,H,W]；收集真实尺寸以决定补齐尺寸。
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            # view_size 是每张图提供的增强视图数，不是批内图片张数。
            view_size = len(img[0])

            # 本批图片大小可能不同，先按最大高宽建零张量，再拷入每张图。
            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            # bi=batch index，vi=view index；把原图片拷到零张量左上角。
            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            # 仍保留“多视图列表”结构；vilt_module.py 中 batch['image'][0]
            # 取的是第 0 个视图，不是 batch 的第 0 张图片。
            dict_batch[img_key] = new_images

        # text/false_text_0 都需做 token 化 batch 和 MLM 目标构造。
        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            # 两层列表展平，便于一次调用 MLM collator；之后按每个 txt_key 切回。
            flatten_encodings = [e for encoding in encodings for e in encoding]
            # mlm_collator 会产生被遮过的 input_ids 与目标 labels。
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                # 本次 txt_key 对应的 batch_size 行从展平结果中切出。
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                # input_ids 保存原词，mlm_ids 保存遮过词；两者用途不同。
                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                # 普通未遮词输入不参与 MLM，故普通 labels 全部 -100。
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
