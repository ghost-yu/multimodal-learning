# 教学版：原文件 ViLT/vilt/modules/vilt_module.py。只增加中文注释，保留原可执行结构。
# B=批量，T=文字 token 数，P=图像 token 数，D=config["hidden_size"]；infer 的 batch 是字典。
# import：torch 张量；nn 神经网络层；pl 是 Lightning 训练框架；vit 是本项目视觉模块别名。
# BertConfig 只储存结构参数，BertEmbeddings 把整数词 ID 变成向量；heads 是任务头，objectives 是损失函数。
# torch 是 PyTorch 主库：load 读检查点，zeros_like/full_like 建与已有张量同形状的索引，cat 拼接。
import torch
# nn（neural network）提供 Embedding、Linear、LayerNorm、GELU、Sequential 等带参数的层。
import torch.nn as nn
# pl 是 PyTorch Lightning：LightningModule 约定训练、验证、测试及优化器的回调函数。
import pytorch_lightning as pl
# vit 是本项目 vision_transformer.py 模块别名；getattr(vit, 配置里的字符串) 动态选模型构造函数。
import vilt.modules.vision_transformer as vit

# BertConfig 是配置，不是训练好的 BERT；BertEmbeddings 把 token ID/位置 ID 转成文字向量。
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
# heads 放分类/预测头，objectives 放各任务损失，vilt_utils 放任务选择、指标、优化器计划。
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        # config 是普通 Python 字典，例如 config["hidden_size"] 是每个 token 的向量宽度 D。
        super().__init__()
        # Lightning 把 __init__ 的 config 记录为 self.hparams.config，后续 infer/training_step 再读取它。
        self.save_hyperparameters()

        bert_config = BertConfig(
            # vocab_size=词表里可查多少种 token；hidden_size=D，文字/图片最终必须同宽才能拼接。
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            # 这两个字段写入 BERT 配置；此处实际使用的是 BertEmbeddings，视觉 block 来自 vit。
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            # MLP 中间宽度=D×倍率；max_position_embeddings 限制文字位置表长度。
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            # 两个 drop_rate 分别对应隐藏激活与注意力概率的随机丢弃率。
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        # 把文本 token ID [B,T] 转成向量序列 [B,T,D]；此处还没有图文融合。
        self.text_embeddings = BertEmbeddings(bert_config)
        # Module.apply 会递归访问子层并执行初始化函数，不是模型的前向计算。
        self.text_embeddings.apply(objectives.init_weights)

        # 类型 0 表示文字，类型 1 表示图片；加到 token 向量上以标明来源。
        # 2 表示文字/图像两种类型；每种类型查得一个 D 维可训练向量，加到对应 token 上。
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            # 空路径说明未提供本任务检查点。getattr 依据 config["vit"] 的名字从 vit 模块找构造器。
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            # 有本任务 checkpoint 时先建未预装视觉权重的骨干，下面再 load_state_dict。
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        # Pooler 把联合序列第 0 个 token 转为整对图文的向量 [B,D]。
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            # MLM=Masked Language Modeling；MLMHead 把每个文字位置 [D] 转成词表 V 个分数。
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            # ITM=Image Text Matching；ITMHead 预测这一对图文是否匹配（两类分数）。
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            # MPP=Masked Patch Prediction；只有配置启用时才建 patch 预测头。
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            # torch.load 读序列化 checkpoint；map_location=cpu 避免文件原 GPU 不可用。
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            # Lightning 检查点里 state_dict 才是“参数名→张量”映射，ckpt 自身还有其他字段。
            state_dict = ckpt["state_dict"]
            # strict=False 允许下游任务头与预训练头不同，不代表任意维度不匹配都可忽略。
            self.load_state_dict(state_dict, strict=False)

        # hs 是 hidden size 的短名，后面各任务头重复用它而无需反复读字典。
        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            # vs=VQA v2 的答案标签类别数；题目+图片的全局向量最终映射成 vs 个答案分数。
            vs = self.hparams.config["vqav2_label_size"]
            # Sequential 按给定顺序调用各层，形状 [B,hs]→[B,2hs]→[B,vs]。
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            # NLVR2 输入两张图；分类头接受拼接后的 [B,2hs] 并输出真假 2 类。
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            # 旧类型表只有文字 0、图片 1；先保存旧权重以扩展到第二张图类型 2。
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            # 第 2 张图的初始类型向量复制自第 1 张图，随后可独立训练。
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            # IRTR=Image Retrieval/Text Retrieval。rank_output 给每对图文一个排序分数 [B,1]。
            self.rank_output = nn.Linear(hs, 1)
            # 取 ITM 分类器中“匹配”类别那一行作为排序层初值；[1:,:] 保留二维形状。
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            # margin 是后续排序损失使用的间隔常数，不是神经网络层。
            self.margin = 0.2
            for p in self.itm_score.parameters():
                # 关闭 ITM 头梯度，避免检索微调又更改用来初始化排序头的分类参数。
                p.requires_grad = False

        # 根据配置建训练/验证指标对象；current_tasks 稍后由 set_task 填入本轮任务名。
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            # 仅测试时要等任务头全部构造完再载入 checkpoint，确保对应头存在。
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
        # infer 在训练/验证也会用；它只做特征抽取，不直接计算损失。
        # batch 是 collate() 产出的字典；mask_text/mask_image 决定是否用遮盖版输入。
        # image_token_type_idx=1 通常指第一张图；可传 2 处理第二张图。
        # image_embeds/image_masks 若外部已算好，可以跳过重复 visual_embed。
        # f 字符串把数字插入键名；例如 idx=1 检查 image_0，供多图任务选择某一张图。
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        # Python 条件表达式：mask_text=True 取遮词输入的 _mlm 后缀，否则空字符串。
        do_mlm = "_mlm" if mask_text else ""
        # text_ids 是整数词 ID [B,T]，来自 collate 整理后的 batch，不是文字的浮点特征。
        text_ids = batch[f"text_ids{do_mlm}"]
        # labels 是目标答案：MLM 时未遮位置常为 -100；不要与模型输入的 ids 混淆。
        text_labels = batch[f"text_labels{do_mlm}"]
        # masks 是有效位 [B,T]，1 为真实文字位置、0 为补齐的 PAD。
        text_masks = batch[f"text_masks"]
        # text_ids 是整数矩阵 [B,T]，embedding 后是浮点向量 [B,T,D]。
        # embeddings 查表后得 [B,T,D]；embeds 是 embedding 的缩写，与下方最终 text_feats 不同。
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            # 两者都未由调用方给出时，才从图片重新计算；若只给其中一个，原实现会走 else，
            # 后续可能在加类型 embedding 时出错，所以外部复用特征应两个一起传。
            # batch[imgkey] 是图片视图列表；[0] 取第一个视图，不代表 batch 中第一张图片。
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
                # max_image_len 控制最多保留多少图像 token；mask_it=True 用于遮 patch 目标。
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            # 复用外部预先算好的 image_embeds/image_masks 时，此处没有 patch 索引/标签可返回。
            patch_index, image_labels = (
                None,
                None,
            )

        # zeros_like(text_masks) 生成类型 ID 0，[B,T]→查 nn.Embedding 得 [B,T,D]；
        # full_like(image_masks,idx) 生成图像类型 ID，[B,P]→[B,P,D]。
        # Python 多变量赋值右侧先求完两个结果，再分别写回 text_embeds、image_embeds。
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        # 沿“token 数”这一维拼接，而非沿特征维拼接：
        # [B,T,D] + [B,P,D] -> [B,T+P,D]。
        # co=combined：在 token 维拼成 [B,T+P,D]。若错用 dim=-1，会拼特征宽度而非序列长度。
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # 特征拼接后，padding mask 也必须以同样顺序拼接。
        # masks 顺序必须与 co_embeds 相同；1 表示有效 token，0 表示 padding。
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        # x 是后续逐层更新的工作变量；起点是原始拼接向量 [B,T+P,D]。
        x = co_embeds

        # 这里图文 token 已在同一个 x 中，因此注意力可让词关注图像 patch。
        # enumerate 同时给层号 i 和层实例 blk；每次循环把上一层的 x 交给下一层。
        for i, blk in enumerate(self.transformer.blocks):
            # blk(x,mask) 返回更新的 token 序列和注意力；_attn 前导 _ 表示本函数暂不使用。
            # 每循环一次 x 都被重新赋值：第 i+1 层读取的是第 i 层的输出。
            x, _attn = blk(x, mask=co_masks)

        # 所有 block 后做最后归一化；x 仍 [B,T+P,D]，只是内容已融合。
        x = self.transformer.norm(x)
        # 用原文字长度作为切分点，从融合序列中拆回两类 token 特征。
        text_feats, image_feats = (
            # 前 T 个位置来自文字，但已通过注意力看过图片，不是纯文本特征。
            x[:, : text_embeds.shape[1]],
            # 后 P 个位置来自图片，切分点必须与上面 cat 的文字长度相同。
            x[:, text_embeds.shape[1] :],
        )
        # pooled CLS 汇总整个图文输入，适合 ITM/VQA 等句级任务。
        # pooler 取融合后的第 0 个 token 做整对图文表示，常给 ITM/VQA 分类头使用。
        cls_feats = self.pooler(x)

        # ret=return 字典：不同损失函数按键读取自己需要的特征和标签。
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            # raw_cls_feats 未经过 pooler，和 cls_feats 的数值通常不同。
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
            # current_tasks 为空时只抽特征；dict.update 把 infer 返回的键值合入 ret。
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            # compute_mlm 返回字典；dict.update 把其中 mlm_loss 等键值并入已有 ret。
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            # MPP 遮图块预测；返回的 mpp_loss 会在 training_step 与其他损失相加。
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            # ITM 判断真/假图文；WPA 用词与 patch 的对齐距离辅助训练。
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            # VQA 用问题文字+图片预测答案类别。
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            # NLVR2 比较一句描述是否符合两张图片。
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            # IRTR 对图文候选打排序分数，常用于检索微调。
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        # Lightning 每训练一个 batch 自动调用。batch_idx 是该批序号，这里未使用。
        # 当前数据集对应哪些训练目标，由配置决定。
        vilt_utils.set_task(self)
        output = self(batch)
        # self(batch) 通过 nn.Module.__call__ 调 forward；output 是按任务汇总的结果字典。
        # 将各目标返回字典里名字带 loss 的张量求和，交给 Lightning 反传。
        # items() 遍历键和值；只取键名含 loss 的张量求和，Lightning 对总损失反传。
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        # 一个训练 epoch 结束时由 Lightning 调用；outs 是各 batch 的返回值集合。
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        # 验证阶段仍要先设任务再计算输出；指标对象在 objectives 内更新。
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        # 测试阶段先执行模型，再只整理需要写出或汇总的测试结果。
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        # 从 load_path 提取检查点文件名：split('/') 取末段，[:-5] 去掉常见 .ckpt 后缀。
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        # Lightning 初始化训练时调用，set_schedule 返回优化器及学习率调度配置。
        return vilt_utils.set_schedule(self)
