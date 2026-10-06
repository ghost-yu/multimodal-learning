# torch 是 PyTorch 主包；本文件用 torch.load 读检查点、torch.cat 拼 token 序列。
import torch
# nn = neural network；Embedding 是查表层，Linear/LayerNorm/GELU 组成任务分类头。
import torch.nn as nn
# pl = PyTorch Lightning；LightningModule 在训练时由 Trainer 调用 training_step 等钩子。
import pytorch_lightning as pl
# vit 是本项目 vision_transformer.py 的短别名；不是 pip 安装的独立 vit 包。
import vilt.modules.vision_transformer as vit

# BertConfig 只记录 BERT 式嵌入参数；BertEmbeddings 把词 ID/位置变为浮点向量。
# 这里没有下载或直接使用一整套预训练 BERT 编码器。
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
# heads 定义分类头；objectives 算各任务损失；vilt_utils 管任务、指标和学习率。
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    # ViL = Vision and Language（视觉与语言）；SS 是原作者类名后缀。
    # 继承 LightningModule 后，Trainer 会按规定名字寻找 training_step/configure_optimizers。
    def __init__(self, config):
        # config 是 Sacred 传来的 Python 字典；键名在 vilt/config.py 中定义。
        super().__init__()
        # 将 __init__ 入参保存为 self.hparams，后续可写 self.hparams.config 重新取配置。
        self.save_hyperparameters()

        # 这里先造文字嵌入层的规格，不会由 BertConfig 自动创建完整 BERT 模型。
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        # 把文本 token ID [B,T] 转成向量序列 [B,T,D]；此处还没有图文融合。
        self.text_embeddings = BertEmbeddings(bert_config)
        # Module.apply 会递归访问子层，调用项目自己的初始化函数；不是前向推理。
        self.text_embeddings.apply(objectives.init_weights)

        # 类型 0 表示文字，类型 1 表示图片；加到 token 向量上以标明来源。
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # load_path 为空说明没有 ViLT 检查点；此时尝试使用 ViT 的视觉初始化权重。
        # getattr(vit, 名字) 按配置字符串找到构造函数，最后一对括号真正创建模型。
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            # 后面会加载完整 ViLT 检查点，因此无需此处另行下载 ViT 初始化权重。
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        # pooler（池化头）取融合序列第 0 个文字 CLS token，输出整对图文的 [B,D] 向量。
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # loss_names 是任务名到权重/开关的字典；大于 0 才创建相应可训练任务头。
        if config["loss_names"]["mlm"] > 0:
            # MLM = masked language modeling，遮住词后预测词表中的原词。
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            # ITM = image-text matching，用 [B,D] 融合向量给真/假两类分数 [B,2]。
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            # MPP = masked patch prediction，遮住部分图像 patch 后预测其颜色统计。
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            # ckpt = checkpoint，磁盘检查点字典；先载入 CPU 避免原 GPU 设备不可用。
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            # state_dict 是“层参数名 -> 参数张量”；它本身不是可直接调用的模型。
            state_dict = ckpt["state_dict"]
            # strict=False 容许下游任务头与预训练检查点不完全同名，不等于忽略所有错误。
            self.load_state_dict(state_dict, strict=False)

        # hs = hidden size，共享 Transformer 每个 token 向量的宽度 D；默认配置常为 768。
        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            # vs = VQA label size，候选答案总数；最终分类器输出 [B,vs]。
            vs = self.hparams.config["vqav2_label_size"]
            # Sequential 会按列出的顺序执行：扩维 -> 归一化 -> 激活 -> 答案打分。
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            # NLVR2 用同一文字分别配两张图，两个 [B,hs] 向量拼成 [B,2*hs]。
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            # 原来只有文字类型 0 和图像类型 1；NLVR2 需给第二张图增加类型 2。
            # emb_data 暂存旧查表权重；.data 是原仓库的底层参数访问写法。
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            # 复制旧文字/图像向量；新增第二图类型先复制第一图类型的初值。
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            # IR/TR = image/text retrieval；每对图文得到一个用于排序的标量分数。
            self.rank_output = nn.Linear(hs, 1)
            # 用已训练的 ITM“真配对”分类权重初始化排序头。
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            # margin 是排序时希望正例领先负例的最小分数间隔。
            self.margin = 0.2
            for p in self.itm_score.parameters():
                # p 是 ITM 分类头中的当前 Parameter；冻结它，避免检索训练更新旧头。
                p.requires_grad = False

        # set_metrics 创建训练/验证用的准确率与损失记录对象。
        vilt_utils.set_metrics(self)
        # current_tasks 是当前 batch 要做的任务名列表，set_task 后会更新。
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            # 纯测试时要在所有下游头都已创建后加载检查点，才能装入这些头的参数。
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        # infer = inference/特征提取；训练目标也调用它，函数名不表示一定关闭梯度。
        # batch 来自 BaseDataset.collate，是键 -> 张量/列表的字典。
        # image_token_type_idx=1 为第一张图；NLVR2 第二张图用 2。
        # image_embeds/image_masks 非 None 时允许复用已算过的图片 token。
        if f"image_{image_token_type_idx - 1}" in batch:
            # f 字符串把索引写进键：idx=1 时键名为 image_0，idx=2 时 image_1。
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            # 普通单图任务由 collate 提供 image 键；imgkey 只是键名字符串。
            imgkey = "image"

        # do_mlm 是键名后缀：mask_text=True 读 text_ids_mlm/labels_mlm。
        do_mlm = "_mlm" if mask_text else ""
        # text_ids 为整数词编号 [B,T]；text_labels 是监督标签，不送入嵌入层。
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        # text_masks 为 [B,T]，真实 token 位置 1、补齐位置 0，后面供注意力用。
        text_masks = batch[f"text_masks"]
        # text_ids 是整数矩阵 [B,T]，embedding 后是浮点向量 [B,T,D]。
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            # batch[imgkey] 是多种图片变换的列表；[0] 取第一种视图，保留整批 B 张图。
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            # 图片切成 patch，并返回 patch 向量及有效位置 mask；
            # patch_index/image_labels 供某些预训练目标使用。
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            # 调用者直接给了图片 token 时，不再切 patch；因此没有原始 patch 标签/索引。
            patch_index, image_labels = (
                None,
                None,
            )

        # zeros_like(text_masks) 造 [B,T] 的 0 索引；查表得文字类型向量 [B,T,D]。
        # full_like(image_masks, idx) 造 [B,P] 的图像类型索引；查表得 [B,P,D]。
        # 两侧仍叫 *_embeds，但此时已含模态类型信息；形状不变。
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        # 沿“token 数”这一维拼接，而非沿特征维拼接：
        # [B,T,D] + [B,P,D] -> [B,T+P,D]。
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # 特征拼接后，padding mask 也必须以同样顺序拼接。
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        # x 是当前层的融合序列 [B,T+P,D]；每遍历一层就重新赋值一次。
        x = co_embeds

        # 这里图文 token 已在同一个 x 中，因此注意力可让词关注图像 patch。
        for i, blk in enumerate(self.transformer.blocks):
            # i 是第几层，blk 是该层 Block；_attn 为返回的注意力矩阵，这里不用。
            x, _attn = blk(x, mask=co_masks)

        # 所有 Transformer Block 完成后再做最终 LayerNorm，形状仍 [B,T+P,D]。
        x = self.transformer.norm(x)
        # 用原文字长度作为切分点，从融合序列中拆回两类 token 特征。
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        # pooled CLS 汇总整个图文输入，适合 ITM/VQA 等句级任务。
        cls_feats = self.pooler(x)

        # ret = return 字典；保留融合特征、标签和索引，供不同 objective 各取所需。
        # raw_cls_feats 是未经 Pooler 的 x[:,0]，与 cls_feats 的数值不同。
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        # infer() 只产出特征；forward() 再依当前任务调用对应损失函数。
        ret = dict()
        if len(self.current_tasks) == 0:
            # 不选择任何任务时只返回编码特征，适合直接做特征读取。
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        # batch_idx 是本 epoch 中的 batch 编号，当前实现并未用它参与计算。
        # 当前数据集对应哪些训练目标，由配置决定。
        vilt_utils.set_task(self)
        output = self(batch)
        # 将各目标返回字典里名字带 loss 的张量求和，交给 Lightning 反传。
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        # outs 是 Lightning 收集的本轮训练输出；此处只做指标汇总。
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        # 验证也设置任务并算输出，但不返回训练损失给优化器。
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        # 测试时 VQA 需要把答案分数变成可提交的预测结构。
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        # model_name 用权重文件名去掉后缀得到；outs 是所有测试 batch 的返回值。
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        # Lightning 调用这个钩子拿优化器与学习率计划；细节在 vilt_utils.set_schedule。
        return vilt_utils.set_schedule(self)
