# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

# os 是 Python 标准库：path.join 组合目录与文件名，避免手工拼接分隔符；路径不是网络地址。
import os
# json 是标准库：loads 把一段 JSON 字符串解析成字典，dumps 把字典变为 JSON 字符串。
# JSONL=JSON Lines，每一行独立存一个 JSON 对象；本项目索引记录图片路径、词编号与任务标签。
import json
# random 是标准随机库：random() 生成 [0,1) 浮点数，randint(a,b) 生成两端都包含的整数。
# 这里决定遮盖位置与替代方式，和模型 Parameter 初始化是不同的随机来源。
import random
# torch=PyTorch，Dataset 规定取样接口，DataLoader 将样本组成批次；这里不负责反向传播。
import torch
# glob 是标准库，用通配符找到本地文件；例如 *.json 匹配所有 json 文件。
import glob
# collections 是标准库：defaultdict(list/set) 在新键出现时自动建容器，Counter 统计元素次数。
from collections import defaultdict, Counter
# torchvision 是 PyTorch 图像工具库，transforms 构造增强/转张量/归一化处理链。
# 常见 Compose 顺序执行、ToTensor 把 PIL 图像变 [3,H,W] 浮点、Normalize 按通道做 (x-mean)/std。
from torchvision import transforms
# default_loader 根据文件路径读取图像（通常 PIL RGB），不是返回已放 GPU 的 Tensor。
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# timm 提供图像模型与增强工具；下方四个常量是不同预处理配方的每通道均值/标准差，不要混用。
# RandomResizedCropAndInterpolation 随机裁剪再缩放；Interpolation 表示选择插值方法。
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

# utils/glossary/randaug 都是本项目文件，区别于上方外部安装库。
import utils
# normalize_word 统一答案词的写法（如数字、大小写与标点相关规则），用于 VQA 数据处理。
from glossary import normalize_word
# RandomAugment 实现随机图像增强；Augment=增强，用变换后的图片帮助训练，不改变文字监督含义。
from randaug import RandomAugment


class BaseDataset(torch.utils.data.Dataset):
    # Base=通用基类；Dataset 约定 __len__ 返回样本数、__getitem__(index) 返回一条样本。
    # 子类通过不同索引文件与任务字段提供 VQA/NLVR2/检索/caption 数据，本类不创建模型。
    def __init__(
        self, data_path, split, transform, 
        tokenizer, num_max_bpe_tokens, task=None,
    ):
        # split=数据划分（train/val/test），不是 multiway 中的模态切分边界。
        # 子类实现 get_index_files；BaseDataset 的占位实现本身不可直接用于这些任务。
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        # num=数量，max=最大，BPE=Byte Pair Encoding（字节对编码）历史命名。
        # 本任务具体 tokenizer 用 SentencePiece；这个字段表示总序列上限，含 BOS/EOS 与补齐。
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    # line 是一个文本行；loads 解析成字典，items 汇总所有索引记录，暂时没有读取全部图像像素。
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        # BOS=beginning of sequence，EOS=end of sequence，PAD=padding；id 是词表整数编号。
        # 从实际 tokenizer 读取，不能照搬 CLIP 的 49406/49407。
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        # image_path 来自索引的相对路径；与 data_path 合成文件位置，再读图和处理。
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        # segment=一段文本。正式索引预处理通常已经把字符串转成整数 token_id 列表。
        # 注意字符串分支只调用 tokenize，得到子词字符串而未 convert_tokens_to_ids；不能假设任意 raw 字符串直接可用。
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            # [:] 创建列表浅副本，以免截断/增加标记时改到 self.items 中长期保存的原列表。
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        # 先截到 max_len-2，为开始/结束词各留一项；+ 拼 Python 列表，不是对编号做数值加法。
        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        # [0]*n 重复列表元素 n 次；0=真实位置，1=padding，应屏蔽。和 CoCa 的 text_valid 布尔语义相反。
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        # 返回三元 tuple：补齐到 max_len 的编号列表、等长 padding 列表、未补齐的真实长度。
        # 目前还是 Python list；DataLoader 的 collate 负责按项目规则转换成批次 Tensor。
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        # data 是一条样本字典；键 image/language_tokens/padding_mask 与 Handler 的参数名对应。
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


def _make_retrieval_coco_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                for sent in item["sentences"]:
                    tokens = tokenizer.tokenize(sent["raw"])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                            "image_path": image_path, 
                            "text_segment": token_ids, 
                            "image_id": len(image_counter), 
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
        (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_retrieval.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


def _make_captioning_coco_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                if item["split"] in ["train", "restval"]:
                    for sent in item["sentences"]:
                        tokens = tokenizer.tokenize(sent["raw"])
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        items.append({
                                "image_path": image_path, 
                                "text_segment": token_ids, 
                                "image_id": item["cocoid"], 
                        })
                else:
                    items.append({
                                "image_path": image_path, 
                                "text_segment": None, 
                                "image_id": item["cocoid"], 
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
        (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_captioning.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


def _make_nocaps_dataset_index(
        data_path,  
        split="val", 
):
    if split == "val":
        json_file = "nocaps_val_4500_captions.json"
    elif split == "test":
        json_file = "nocaps_test_image_info.json"
    nocaps_split_json_file = os.path.join(data_path, json_file)
    items = []
    image_counter = set()
    print("read %s" % nocaps_split_json_file)
    with open(nocaps_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            image_path = os.path.join(split, item["file_name"])
            items.append({
                "image_path": image_path, 
                "text_segment": None, 
                "image_id": item["id"], 
            })

            if image_path not in image_counter:
                image_counter.add(image_path)

    print("Find %d images and %d image-text pairs for nocaps dataset %s split !" % \
        (len(image_counter), len(items), split))
    index_file = os.path.join(data_path, "nocaps.%s.jsonl" % split)
    _write_data_into_jsonl(items, index_file)


class NLVR2Dataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("nlvr2.train.index.jsonl", )
        elif split == "val":
            return ("nlvr2.dev.index.jsonl", )
        elif split == "test":
            return ("nlvr2.test-P.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["label"] = self.items[index]["label"]
        return data

    @staticmethod
    def __preprocess_json(preifx, json_file, tokenizer, index_file):
        items = []
        with open(json_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                path = os.path.join(preifx, str(data["directory"])) if "directory" in data else preifx
                path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
                tokens = tokenizer.tokenize(data["sentence"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                items.append({
                    "image_path": path + "-img0.png",
                    "image2_path": path + "-img1.png",
                    "text_segment": token_ids,
                    "label": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"], 
                })
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, nlvr_repo_path):
        cls.__preprocess_json(
            preifx="images/train", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/train.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("train")[0]), 
        )
        cls.__preprocess_json(
            preifx="dev", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/dev.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("val")[0]), 
        )
        cls.__preprocess_json(
            preifx="test1", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/test1.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("test")[0]), 
        )


class ImageNetDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("imagenet.train.index.jsonl", )
        elif split == "val":
            return ("imagenet.val.index.jsonl", )
        elif split == "test":
            return ("imagenet.val.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["label"] = item["label"]
        return data
    
    @staticmethod
    def _find_classes(dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _make_imagenet_index(data_path, index_path, data_path_prefix, class_to_idx, split):
        items = []
        index_file = os.path.join(index_path, f"imagenet.{split}.index.jsonl")
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(data_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    path = path.replace(data_path_prefix, "")
                    items.append({
                        "image_path": path,
                        "label": class_index,
                    })

        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, train_data_path, val_data_path, index_path):
        data_path_prefix = train_data_path[:[x[0]==x[1] for x in zip(train_data_path, val_data_path)].index(0)]
        classes, class_to_idx = cls._find_classes(train_data_path)
        cls._make_imagenet_index(
             data_path=train_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
             class_to_idx=class_to_idx, split="train",
        )
        cls._make_imagenet_index(
             data_path=val_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
             class_to_idx=class_to_idx, split="val",
        )


class VQAv2Dataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        ans2label_file = os.path.join(data_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = i
                label2ans.append(ans)
        
        self.ans2label = ans2label
        self.label2ans = label2ans

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("vqa.train.jsonl", "vqa.trainable_val.jsonl")
        elif split == "val":
            return ("vqa.rest_val.jsonl", )
        elif split == "test":
            return ("vqa.test.jsonl", )
        elif split == "test-dev":
            return ("vqa.test-dev.jsonl", )            
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        if "labels" in self.items[index] and len(self.items[index]["labels"]) > 0:
            labels = [0.] * len(self.label2ans)
            for l, s in zip(self.items[index]["labels"], self.items[index]["scores"]):
                labels[l] = s
            data["labels"] = torch.FloatTensor(labels)
        else:
            data["qid"] = self.items[index]["qid"]
        return data

    @staticmethod
    def get_score(occurences):
        if occurences == 0:
            return 0.0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1.0

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, annotation_data_path):
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]

        with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        for split, questions in zip(
            ["train", "val", "test", "test-dev"],
            [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                tokens = tokenizer.tokenize(question_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert q["question_id"] not in _annot[q["image_id"]]
                _annot[q["image_id"]][q["question_id"]] = {
                    "question": question_text, 
                    "token_ids": token_ids, 
                }

            annotations[split] = _annot

        all_major_answers = list()

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            # _annot = annotations[split]
            for q in annots:
                all_major_answers.append(q["multiple_choice_answer"])

        all_major_answers = [normalize_word(word) for word in all_major_answers]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q in annots:
                answers = q["answers"]
                answer_count = {}
                for answer in answers:
                    answer_ = answer["answer"]
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in ans2label:
                        continue
                    labels.append(ans2label[answer])
                    score = cls.get_score(answer_count[answer])
                    scores.append(score)

                assert "labels" not in _annot[q["image_id"]][q["question_id"]]
                assert "question" in _annot[q["image_id"]][q["question_id"]]
                _annot[q["image_id"]][q["question_id"]]["labels"] = labels
                _annot[q["image_id"]][q["question_id"]]["scores"] = scores

        for split in ["train", "val"]:
            filtered_annot = dict()
            for ik, iv in annotations[split].items():
                new_q = dict()
                for qk, qv in iv.items():
                    if len(qv["labels"]) != 0:
                        new_q[qk] = qv
                if len(new_q) != 0:
                    filtered_annot[ik] = new_q
            annotations[split] = filtered_annot

        split2items = {}
        for split in ["train", "val", "test", "test-dev"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014",
                "val": "val2014",
                "test": "test2015",
                "test-dev": "test2015",
            }[split]
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            random.shuffle(paths)
            annot_paths = [path for path in paths \
                if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                print("all images have caption annotations")
            else:
                print("not all images have caption annotations")
            print(len(paths), len(annot_paths), len(annot))

            items = []
            for path in annot_paths:
                iid = int(path.split("/")[-1].split("_")[-1][:-4])
                _annot = annotations[split][iid]
                for qid in _annot:
                    q = _annot[qid]
                    if split in ["train", "val"]:
                        labels = q["labels"]
                        scores = q["scores"]
                    else:
                        labels, scores = [], []

                    items.append({
                        "image_path": os.path.join(split_name, path.split('/')[-1]), 
                        "text_segment": q["token_ids"], 
                        "labels": labels, 
                        "scores": scores, 
                        "qid": qid, 
                    })
            split2items[split] = items

            _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

        # Following ViLT, we use 1000 images of the original val set as the final val set        
        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)
        
        print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_id in enumerate(val_images):
            if i < 1000:
                rest_val += val_image2items[image_id]
            else:
                trainable_val += val_image2items[image_id]
        
        _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
        _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))


class RetrievalDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["image_id"] = self.items[index]["image_id"]
        return data

    @staticmethod
    def make_flickr30k_dataset_index(data_path, tokenizer, karpathy_path):

        with open(os.path.join(karpathy_path, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        captions = captions["images"]
        split2items = defaultdict(list)
        split2images = defaultdict(set)

        for each_item in captions:
            image_path = os.path.join("flickr30k-images", each_item["filename"])
            split = each_item["split"]

            for text_segment in each_item["sentences"]:
                tokens = tokenizer.tokenize(text_segment["raw"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                split2items[split].append({
                    "image_path": image_path, 
                    "text_segment": token_ids, 
                    "image_id": len(split2images[split]), 
                })

            assert each_item["filename"] not in split2images[split]
            split2images[split].add(each_item["filename"])

        for split in split2items:
            print("%d images and %d image-text pairs!" % (len(split2images[split]), len(split2items[split])))
            _write_data_into_jsonl(split2items[split], os.path.join(data_path, "flickr30k.%s.jsonl" % split))

    @staticmethod
    def make_coco_dataset_index(data_path, tokenizer):
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")


class CaptioningDataset(BaseDataset):

    def __init__(self, data_path, split, transform, 
                tokenizer, num_max_bpe_tokens, task, mask_prob):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task, 
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("coco_captioning.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def _get_mask_token(self, token):
        # token 是原整数词编号，本函数为“已选为监督目标的位置”选择实际输入内容。
        p = random.random()
        # 80% 换 MASK、10% 保留原词、10% 换随机词；保留原词仍可能需要预测，因为监督掩码另行记录。
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        # tokens 是要修改的编号副本；num_tokens 含 BOS/EOS、不含右侧 PAD；mask_prob 是选择监督位置的比例。
        bool_masked_pos = [0] * len(tokens)
        # +0.5 再 int 对正数近似四舍五入；min 限制最多 num_tokens-1 个，避免选择 BOS。
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            # randint 含端点：从位置 1 到 EOS 的位置都可选；位置 0 的 BOS 和后续 PAD 不会选。
            # EOS 也可成为目标，模型才能学会什么时候结束描述。
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                # 同一位置不重复计数；原答案另存在 language_tokens 中，这里改的是 masked_tokens 的副本。
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
            # language_tokens=完整监督答案；masked_tokens=实际输入，必须复制再遮盖以免把正确答案也改掉。
            masked_tokens = language_tokens[:]
            masked_tokens, language_masked_pos = \
                self._masking_on_text_tokens(masked_tokens, num_tokens, self.mask_prob)
            data["language_tokens"] = language_tokens
            data["masked_tokens"] = masked_tokens
            data["language_masked_pos"] = language_masked_pos
            data["padding_mask"] = padding_mask
        return data

    @staticmethod
    def make_coco_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")

    @staticmethod
    def make_nocaps_captioning_dataset_index(data_path):
        _make_nocaps_dataset_index(data_path, split="val")
        _make_nocaps_dataset_index(data_path, split="test")


task2dataset = {
    "nlvr2": NLVR2Dataset, 
    "vqav2": VQAv2Dataset, 
    "flickr30k": RetrievalDataset, 
    "coco_retrieval": RetrievalDataset,  
    "coco_captioning": CaptioningDataset,
    "nocaps": CaptioningDataset,
    "imagenet": ImageNetDataset,
}


# loader=加载器：负责抽取样本、组批、转成批次 Tensor；不创建模型也不计算损失。
# batch_size=每个进程每次取多少样本；num_workers=读取数据的子进程数量，不是 GPU 数量。
# pin_mem=pinned memory（锁页内存）的缩写，配合 GPU 复制；dist_eval=distributed evaluation（分布式评估）。
def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        # 此处 tasks 指分布式进程总数 world_size，不是 VQA/检索等学习任务的种类数。
        num_tasks = utils.get_world_size()
        # rank=进程序号，从 0 开始；每个进程据此分到不同样本。
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        # sampler=抽样器，只给样本索引；DistributedSampler 按进程切分索引。
        # 为使各进程样本数相同，默认可能重复尾部样本；上面的 warning 就提示这种评估影响。
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        # Sequential=顺序：按 0,1,2,... 抽样，便于非分布式评估的稳定顺序。
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    # DataLoader 调用 Dataset.__getitem__ 取得字典，再用下方 collate_fn 合并同名字段。
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        # drop_last=是否丢掉不足一整批的尾部；训练 True、评估 False（由调用者 is_train 决定）。
        drop_last=is_train,
        # collate=整理组批；本地函数把 image 堆成 [B,3,H,W]、整数列表转成 [B,T]。
        # 这不是 PyTorch 默认 collate：默认对普通列表的处理方式不能直接代替这里的约定。
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


# transform=预处理/增强操作链；args 是命令行解析后的命名空间，通过 args.字段名取配置。
def build_transform(is_train, args):
    if args.task in ["imagenet"]:
        return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0), interpolation=args.train_interpolation), 
            # Horizontal=水平，Flip=翻转；默认以 0.5 概率左右镜像，图片输出尺寸不变。
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
                    ]))
        t += [
            # ToTensor 将 PIL RGB 的 0..255 像素转为 float [3,H,W]，通常缩放到 0..1。
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        # t 先是操作列表，随后变为可调用 Compose 对象；Compose 依次执行列表中的操作。
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            # 二元尺寸会直接缩放到正方形；与给单个整数、保持宽高比例的 Resize 不同。
            # interpolation=3 是这里旧 PIL/torchvision 接口的 bicubic（双三次插值）编号。
            transforms.Resize((args.input_size, args.input_size), interpolation=3), 
            # ToTensor 将 PIL RGB 的 0..255 像素转为 float [3,H,W]，通常缩放到 0..1。
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def build_imagenet_transform(is_train, args):
    # im=image 的缩写；这里的变量是布尔开关，不是图像本身。小尺寸图片走另一裁剪策略。
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            args.crop_pct = 1.0
        # crop_pct=crop percentage，裁剪保留比例；如 input_size=224、crop_pct=0.875，先缩到 256。
        # int 转为整数；随后 CenterCrop 裁到 224，训练和评估的增强策略不同。
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


# 此处是函数内 import：调用这个函数时才需要加载对应 tokenizer 类。
def get_sentencepiece_model_for_beit3(args):
    # transformers 是 Hugging Face 模型/tokenizer 库；XLM=跨语言模型，RoBERTa 是模型系列名。
    # 这里仅使用 tokenizer，不创建 XLM-R 语言模型；内部依赖 SentencePiece 分词模型文件。
    # tokenize 返回子词字符串；convert_tokens_to_ids 才把字符串变成词表整数编号。
    from transformers import XLMRobertaTokenizer
    # sentencepiece_model 是本地分词模型路径，不能用任意文本文件代替；编号要和 checkpoint 词表匹配。
    return XLMRobertaTokenizer(args.sentencepiece_model)


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    # task2dataset 中的 2 读作 to（映射到）；值是类对象，不是已经加载的数据实例。
    dataset_class = task2dataset[args.task]
    tokenizer = get_sentencepiece_model_for_beit3(args)

    # opt=optional（可选），kwargs=keyword arguments（关键字参数）；按任务额外传构造参数。
    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    # 调用类对象执行构造方法，生成当前任务 Dataset；**opt_kwargs 把字典展开成命名参数。
    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, tokenizer=tokenizer, 
        num_max_bpe_tokens=args.num_max_bpe_tokens, 
        task=args.task, **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
    # hasattr 检查对象是否有该字段，短路 and 保证不存在字段时不会继续访问。
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        # 无显式评估批大小时用训练的 1.5 倍；只是项目默认选择，不保证任意 GPU 都容得下。
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )


# downstream=下游任务，与前面的通用骨干预训练区分；这里返回 DataLoader。
def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        # 注意原仓库这里 val 也传 is_train=True：会使用训练增强/随机抽样/drop_last。
        # 不要因 split 名叫 val 就推断它自动获得完整确定性评估设置；这是待核查的原始行为。
        # 反斜杠是 Python 续行，逗号使返回值成为 (train_loader, val_loader) 二元 tuple。
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=True)
