# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

# math/sys/json 是 Python 标准库：isfinite 检查数值；exit 结束失败进程；JSON 保存评估结果。
import math
import sys
import json
# Iterable=可逐项遍历的对象，如 DataLoader；Optional[T]=T 或 None。这些类型提示不运行模型。
from typing import Iterable, Optional

# torch 是 PyTorch：Tensor 运算、设备转移、自动求导与混合精度接口都来自它。
import torch
# nn=neural network（神经网络）；Module 是模型基类，损失类也属于该命名空间。
import torch.nn as nn
# F=functional（函数式接口），例如 F.softmax；函数本身不以 Module 对象维护可训练参数。
import torch.nn.functional as F

from timm.utils import ModelEma
# accuracy 计算 top-k 分类准确率；ModelEma=Exponential Moving Average（模型指数滑动平均），维护平滑权重。
# 上一行已导入 ModelEma，这里再次导入是原文件写法，注释不为此改变运行代码。
from timm.utils import accuracy, ModelEma
# LabelSmoothing=标签平滑，给非目标类别少量概率；SoftTarget=软目标，适用于 mixup 后的概率标签。
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# datasets 在这里是相邻 datasets.py，不是 Hugging Face 同名数据集库；此函数读取本地 tokenizer。
from datasets import get_sentencepiece_model_for_beit3

# utils 是相邻 utils.py：日志、分布式汇总、任务损失等辅助工具；不是 Python 标准模块。
import utils


class TaskHandler(object):
    # Handler=处理器，负责把当前任务的样本字段转给模型、选损失与汇总评估。
    # 不是新的神经网络；model 负责前向，Handler 管数据/目标接口。
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class NLVR2Handler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_batch(self, model, image, image2, language_tokens, padding_mask, label):
        logits = model(
            image_a=image, image_b=image2, 
            text_description=language_tokens, 
            padding_mask=padding_mask)
        acc = (logits.max(-1)[-1] == label).float().mean()
        return {
            "loss": self.criterion(input=logits, target=label), 
            "acc": acc, 
        }

    def eval_batch(self, model, image, image2, language_tokens, padding_mask, label):
        logits = model(
            image_a=image, image_b=image2, 
            text_description=language_tokens, 
            padding_mask=padding_mask)
        batch_size = language_tokens.shape[0]
        acc = (logits.max(-1)[-1] == label).float().sum(0) * 100.0 / batch_size
        self.metric_logger.meters['acc'].update(acc.item(), n=batch_size)
    
    def after_eval(self, **kwargs):
        print('* Acc {acc.global_avg:.3f}'.format(acc=self.metric_logger.acc))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc"
    

class ImageNetHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif args.label_smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train_batch(self, model, image, label):
        logits = model(image=image)
        return {
            "loss": self.criterion(logits, label), 
        }

    def eval_batch(self, model, image, label):
        logits = model(image=image)
        batch_size = image.shape[0]
        acc1, acc5 = accuracy(logits, label, topk=(1, 5))
        self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    def after_eval(self, **kwargs):
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
            .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc1"


class RetrievalHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.metric_logger = None

    def train_batch(self, model, image, language_tokens, padding_mask, image_id):
        # 模型内部已算 CLIP 式图文对比损失；Handler 交给共用训练循环即可。
        loss, vision_cls, language_cls = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask)
        return {
            "loss": loss, 
        }

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, language_tokens, padding_mask, image_id):
        # 评估时分别编码图和文，缓存向量后可批量比较所有候选。
        vision_cls, _ = model(image=image, only_infer=True)
        _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)

        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())
    
    def after_eval(self, **kwargs):
        image_feats = {}
        for feats, ids in zip(self.image_feats, self.image_ids):
            for i, _idx in enumerate(ids):
                idx = _idx.item()
                if idx not in image_feats:
                    image_feats[idx] = feats[i]
        
        tiids = torch.cat(self.image_ids, dim=0)
        iids = []
        sorted_tensors = []
        for key in sorted(image_feats.keys()):
            sorted_tensors.append(image_feats[key].view(1, -1))
            iids.append(key)

        image_cls_feats = torch.cat(sorted_tensors, dim=0)
        text_cls_feats = torch.cat(self.text_feats, dim=0)

        scores = image_cls_feats @ text_cls_feats.t()
        iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))

        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        
        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]

        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk1 = scores.topk(1, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]

        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
        ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

        eval_result = {
            "tr_r10": tr_r10.item() * 100.0, 
            "tr_r5": tr_r5.item() * 100.0, 
            "tr_r1": tr_r1.item() * 100.0, 
            "ir_r10": ir_r10.item() * 100.0, 
            "ir_r5": ir_r5.item() * 100.0, 
            "ir_r1": ir_r1.item() * 100.0, 
            "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0, 
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"


class VQAHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        # BCE=Binary Cross Entropy（二元交叉熵），逐答案候选处理 VQA 的软分数，不是单类别 softmax。
        # WithLogits 表示输入原始分数，内部稳定地合入 sigmoid；无需先手工 sigmoid。
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.label2ans = None

    def train_batch(self, model, image, language_tokens, padding_mask, labels):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        return {
            # logits/labels=[B,A]，A 为答案候选数；mean 对 B*A 平均，再乘 A 变为“每样本答案项求和的平均”。
            # float 转 float32 保持数值计算精度；它不会像 detach 一样切断求导。
            "loss": self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1], 
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()
        self.metric_logger = metric_logger
        self.label2ans = data_loader.dataset.label2ans

    def eval_batch(self, model, image, language_tokens, padding_mask, labels=None, qid=None):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        batch_size = language_tokens.shape[0]
        if labels is not None:
            scores = utils.VQAScore()(logits, labels) * 100.0
            self.metric_logger.meters['score'].update(scores.item(), n=batch_size)
        else:
            _, preds = logits.max(-1)
            for image_id, pred in zip(qid, preds):
                self.predictions.append({
                    "question_id": image_id.item(), 
                    "answer": self.label2ans[pred.item()], 
                })

    def after_eval(self, **kwargs):
        if len(self.predictions) == 0:
            print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
            return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"
        else:
            return self.predictions, "prediction"


# 【caption详解】Captioning=生成图片描述，Handler=任务处理器；继承 TaskHandler，管理训练目标和生成结果，实际可学习模型仍由参数 model 传入。
class CaptioningHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        self.predictions = []
        # 【caption详解】criterion=损失准则；BertCaptioningLoss 用标签平滑的 KL 散度，可在指定步数后丢弃部分最高损失 token。它不是普通无平滑交叉熵的数值完全等价实现。
        self.criterion = utils.BertCaptioningLoss(args.label_smoothing, args.drop_worst_ratio, args.drop_worst_after)
        # 【caption详解】tokenizer=分词器，负责文字↔整数 token IDs；SentencePiece 的 piece 表示子词片段，不保证一个 token 就是一个完整英文单词。
        self.tokenizer = get_sentencepiece_model_for_beit3(args)
        # 【caption详解】num_beams=beam 数 K，即每张图片同时保留的候选前缀数；K 越大通常越耗计算，不保证描述一定更准确。
        self.num_beams = args.num_beams
        # 【caption详解】max_len=序列最大长度 T，包含特殊 token；BPE=Byte Pair Encoding，字节对编码，配置名字沿用 token 数约定，不是最多英文单词数。
        self.max_len = args.num_max_bpe_tokens
        # 【caption详解】length_penalty=长度归一化指数 α；完成候选的总 log 概率除以长度^α，以调节对短句的偏好。
        self.length_penalty = args.length_penalty
        # 【caption详解】vocab_size=词表大小 V，所有可预测 token 的总数；不是当前句子的长度。
        self.vocab_size = args.vocab_size

    def train_batch(self, model, image, language_tokens, masked_tokens, language_masked_pos, padding_mask, image_id, global_step):
        logits, _ = model(
            image=image, text_ids=masked_tokens, padding_mask=padding_mask, language_masked_pos=language_masked_pos, image_id=image_id)
        # 原文字 language_tokens=[B,T]，mask=[B,T]；布尔索引得被选中位置的答案 [M]。
        # logits=[M,V]，选择顺序必须一致，不能用 CoCa 下一词 [1:] 标签机械代替。
        masked_labels = language_tokens[language_masked_pos.bool()]
        score = torch.max(logits, -1)[1].data == masked_labels
        acc = torch.sum(score.float()) / torch.sum(language_masked_pos)
        return {
            "loss": self.criterion(logits, masked_labels, global_step),
            "acc": acc
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()
        self.metric_logger = metric_logger

    # 【caption详解】eval_batch 对一批图像生成描述；image=[B,3,H,W]，image_id=[B] 为结果关联数据集样本，不能把它当类别标签。model 是外部模型，self 是处理器。
    def eval_batch(self, model, image, image_id=None):
        # 【caption详解】cur_len=current length，初始两位为 [CLS,MASK]；之后表示包括当前位置 MASK 的逻辑历史总长度，不等于每次短输入始终为2的物理长度。
        cur_len = 2
        # 【caption详解】每张图片最终保留最优1条句子；与同时探索 K 条 beam 不同。后面 squeeze(1) 也依赖这项为1。
        num_keep_best = 1
        # 【caption详解】TOPN_PER_BEAM 表示每步总共先取 3K 个候选，给 EOS 结束候选留空间；名称大写是常量习惯，但 topk 实际在合并的 K×V 空间里全局选。
        TOPN_PER_BEAM = 3

        # 【caption详解】size(0) 取图像第0维，即图片数 B；稍后模型批量轴展开为 B×K。
        batch_size = image.size(0)
        # 【caption详解】mask_id 是 MASK 特殊 token 的词表编号；放在尚未知的词位置，让模型预测它。
        mask_id = self.tokenizer.mask_token_id
        # 【caption详解】CLS=classification 特殊标记，当前作为句子开头；在这里不是要求给图片分类。
        cls_id = self.tokenizer.cls_token_id
        # 【caption详解】PAD=padding 填充标记，用于让完成/不同长度结果保持相同张量尺寸。
        pad_id = self.tokenizer.pad_token_id
        # 【caption详解】SEP=separator 分隔标记，当前用作句子结束 EOS；具体数字取决于 tokenizer，不能写死猜测。
        sep_id = self.tokenizer.sep_token_id
        # 【caption详解】EOS=End Of Sequence（序列结束）；列表允许匹配结束词编号，当前只含 SEP。
        eos_token_ids = [sep_id]

        # 【caption详解】torch.full 按尺寸创建所有元素同为某值的张量；cls_ids 的复数表示每个样本各一项 CLS。下方 dtype=long 适合整数词表索引。
        cls_ids = torch.full(
            (batch_size, 1), cls_id, dtype=torch.long, device=image.device
        )
        # 【caption详解】同样为每张图片创建一个 MASK 编号 [B,1]，不是创建布尔注意力 mask。
        mask_ids = torch.full(
            (batch_size, 1), mask_id, dtype=torch.long, device=image.device
        )
        # 【caption详解】cat=concatenate 拼接；dim=1 沿文字长度轴，把两个 [B,1] 组成 [B,2]，开始时每行为 CLS,MASK。
        cur_input_ids = torch.cat([cls_ids, mask_ids], dim=1)
        # 【caption详解】tmp=temporary 暂存；创建 [B,T-1] 的 MASK，占位结果缓冲区尚未生成的每个位置。
        tmp_ids = torch.full(
            (batch_size, self.max_len-1), mask_id, dtype=torch.long, device=image.device
        )
        # 【caption详解】decoding_results 是累计整句缓冲 [B,T]，与只保留2个当前输入 token 的 cur_input_ids 不同。
        decoding_results = torch.cat([cls_ids, tmp_ids], dim=1)
        
        # Expand input to num beams
        # 【caption详解】unsqueeze(1) 插入 beam 轴 [B,1,2]；expand 扩为 [B,K,2]，通常通过共享存储视图，不立即复制每份数据。
        cur_input_ids = cur_input_ids.unsqueeze(1).expand(batch_size, self.num_beams, cur_len)
        # 【caption详解】expand 后步幅可能含0，先 contiguous 复制成连续内存，再 view 折叠 B,K 为 BK；每张图片 K 条 beam 相邻排列。
        cur_input_ids = cur_input_ids.contiguous().view(batch_size * self.num_beams, cur_len)  # (batch_size * num_beams, cur_len)
        # 【caption详解】同样扩展结果缓冲为 [B,K,T]；每条 beam 后面会记录不同词序列。
        decoding_results = decoding_results.unsqueeze(1).expand(batch_size, self.num_beams, self.max_len)
        # 【caption详解】整理为 [BK,T]；原行尾注释写 cur_len，但实际第二维 self.max_len，是缓冲容量，不是当前已生成长度。
        decoding_results = decoding_results.contiguous().view(batch_size * self.num_beams, self.max_len)  # (batch_size * num_beams, cur_len)
        # 【caption详解】图像扩成 [B,K,3,H,W]；size(-3/-2/-1) 分别取原图像通道、高、宽。每条候选共享同一图片内容。
        image = image.unsqueeze(1).expand(batch_size, self.num_beams, image.size(-3), image.size(-2), image.size(-1))
        # 【caption详解】图像整理为 [BK,3,H,W]；contiguous 后实际复制可能增加显存，beam 扩展不是免费计算。
        image = image.contiguous().view(batch_size * self.num_beams, image.size(-3), image.size(-2), image.size(-1))

        # 【caption详解】generated_hyps=已完成的 hypotheses（候选句子）容器列表；每张图片一个容器，活跃 beam 在另一个列表管理。
        generated_hyps = [
            # 【caption详解】BeamHypotheses 是相邻 utils.py 的类，不是 torch API；负责保存完成句子、长度评分和停止判断。
            utils.BeamHypotheses(
                num_keep_best, self.max_len, length_penalty=self.length_penalty, early_stopping=False
            ) for _ in range(batch_size)
        ]
        # scores for each sentence in the beam
        # 【caption详解】beam_scores=[B,K] 存活跃前缀的累计 log 概率；起始第0条为0，下一行抑制重复起点。
        beam_scores = torch.zeros((batch_size, self.num_beams), dtype=torch.float, device=cur_input_ids.device)
        # 【caption详解】仅第0条初始 beam 有效，其余设很小的负数；否则 K 条相同 CLS 前缀会重复占据最优候选。-1e9 是近似负无穷。
        beam_scores[:, 1:] = -1e9
        # 【caption详解】view(-1) 展平 [B,K]→[BK]；-1 让 PyTorch 推导该维长度。
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        # 【caption详解】done 是每张图片是否停止的 Python 布尔列表；_ 是不使用的循环索引。
        done = [False for _ in range(batch_size)]
        # 【caption详解】incremental_state=增量状态缓存，按层存注意力 key/value 张量，避免每步重算整个图片与历史文字。空字典不是 None，所以模型会启用增量路径。
        incremental_state = {}

        # 【caption详解】while 重复生成，cur_len 每轮加1；同时受 done 和最大长度逻辑影响。
        while cur_len <= self.max_len:
            # 【caption详解】本次模型输入始终是2个文字 token，新 MASK 在索引1；因此从输出第1位取分数，而非从逻辑历史长度 cur_len-1 直接索引短输出。
            next_token_idx = 1
            # 【caption详解】padding_masks 在这次当前短输入上全部为0，表示没有 padding；不是标注 MASK token，二者作用不同。
            padding_masks = torch.full(
                cur_input_ids.shape, 0, dtype=torch.long, device=image.device
            )
            # 【caption详解】首轮需要送入复制后的 [BK,3,H,W] 图像，建立视觉缓存。
            input_image = image
            # 【caption详解】后续历史长度不为2时，图片已经在缓存内，不再重新编码；假设此前缓存正常保留。
            if cur_len != 2:
                # 【caption详解】None 触发 BEiT3ForCaptioning.forward 的后续增量路径；不代表模型不知道图片。
                input_image = None

            # 【caption详解】模型返回本轮文字词表 logits [BK,2,V] 和更新缓存；outputs 不是已解码字符串。
            outputs, incremental_state_next = model(
                image=input_image, text_ids=cur_input_ids, language_masked_pos=None,
                padding_mask=padding_masks, text_len=cur_len, incremental_state=incremental_state)
            # 【caption详解】更新当前缓存引用，用于后面按 beam 重排与去掉临时 MASK 的 key/value。
            incremental_state = incremental_state_next

            # assert outputs.shape[1] == token_len
            # 【caption详解】切出每条 beam 的 MASK 位置分数 [BK,V]；scores 现在仍是 logits。
            scores = outputs[:, next_token_idx, :] # (batch_size * num_beams, vocab_size)
            # 【caption详解】F 是 torch.nn.functional；log_softmax 沿词表归一化并取对数，数值稳定，输出 log 条件概率，不是概率本身。
            scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            assert scores.size() == (batch_size * self.num_beams, self.vocab_size)
            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            # 【caption详解】[:,None] 把累计前缀分数 [BK]→[BK,1]，expand_as 广播到 [BK,V]；相加等于把新词概率乘到整句概率再取 log。前导下划线表示暂存，不是特殊语法。
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            # 【caption详解】将每张图片 K 条 beam×V 个词合并为一个候选轴 [B,KV]，下一步在所有父 beam 中共同选优。
            _scores = _scores.view(batch_size, self.num_beams * self.vocab_size)  # (batch_size, num_beams * vocab_size)
            # 【caption详解】topk 返回数值与索引；next_words 是展平空间索引，不是直接词编号。largest=True 取最高 log 概率；sorted=True 降序。KV 至少应容纳3K候选。
            next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * self.num_beams, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * self.num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            # 【caption详解】存本轮所有图片下一步活跃 beam，每项为(累计分数,新词ID,父beam在BK批量中的行号)。
            next_batch_beam = []
            # for each sentence
            # 【caption详解】batch_ex=batch example，当前图片在原始 B 批量中的索引，不是 beam 行号。
            for batch_ex in range(batch_size):
                # if we are done with this sentence
                # 【caption详解】or 短路保持已经结束的状态；is_done 用最优活跃分数判断完成候选是否足够好，.item 将单元素张量转成 Python 数值。
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    # 【caption详解】extend 将 K 个填充条目加入批量，保持尺寸；父行号0是占位，完成样本不应再使用其生成语义。
                    next_batch_beam.extend([(0, pad_id, 0)] * self.num_beams)  # pad the batch
                    continue

                # next sentence beam content
                # 【caption详解】当前单张图片下一轮的活跃候选列表，与 generated_hyps 中已完成句子不同。
                next_sent_beam = []
                # 【caption详解】zip 按位置配对展平索引与分数，按降序逐候选处理；idx/score 是单元素张量。
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    # get beam and word IDs
                    # 【caption详解】// 是整除：展平索引除以 V 得父 beam 在该图中的局部编号，范围0到K-1。
                    beam_id = idx // self.vocab_size
                    # 【caption详解】% 是取余：得到真正的新词编号，范围0到V-1。例如 V=10，idx=17 表示父beam1、词7。
                    word_id = idx % self.vocab_size
                    # end of sentence, or next word
                    # if word_id.item() in eos_token_ids or cur_len + 1 == max_len:
                    # 【caption详解】新词为EOS或到长度边界时保存完成候选；原实现保存当前缓冲前缀，还没有把此次选中的 word_id 写入，所以非EOS在长度边界可能被丢弃，不能按常规算法假定已写入。
                    if (word_id.item() in eos_token_ids and cur_len + 1 <= self.max_len) or (cur_len + 1 == self.max_len):
                        generated_hyps[batch_ex].add(
                            # 【caption详解】batch_ex*K+beam_id 找父候选行；切前 cur_len 项再 clone 固定副本，避免后续缓冲覆盖。此时末位可能仍是 MASK 占位，decode 时依靠 skip_special_tokens 去掉。
                            decoding_results[batch_ex * self.num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        # 【caption详解】未完成候选记住新词与父行号，后续复制父历史并写入该词。append 加入单个元组。
                        next_sent_beam.append((score, word_id, batch_ex * self.num_beams + beam_id))
                    # the beam for next step is full
                    # 【caption详解】活跃候选达到K即停止扫描；已结束候选不占活跃槽。
                    if len(next_sent_beam) == self.num_beams:
                        break

                # update next beam content
                if cur_len + 1 == self.max_len:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == self.num_beams

                if len(next_sent_beam) == 0:
                    # 【caption详解】长度边界没有活跃项时用K条PAD占位；这是维持批量结构，不是K条有效答案。
                    next_sent_beam = [(0, pad_id, 0)] * self.num_beams  # pad the batch
                # 【caption详解】把当前图片K个活跃/占位条目接到全批列表，保证原始图片仍分组排列。
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == self.num_beams * (batch_ex + 1)
            
            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * self.num_beams
            # 【caption详解】Tensor.new 根据旧 beam_scores 的 dtype/device 创建新分数数组；列表推导 x[0] 取每个元组的累计分数。
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            # 【caption详解】按旧整数输入的dtype/device创建新词ID [BK]；x[1] 是元组第二项。
            beam_words = cur_input_ids.new([x[1] for x in next_batch_beam])
            # 【caption详解】beam_idx 存父beam行号 [BK]，用于重排历史与缓存；可以有重复，因为同一父句可扩出多个不同词。
            beam_idx = cur_input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            # 【caption详解】高级索引按父行号复制/重排当前输入，行数仍BK；后面会被新词+MASK短输入覆盖。
            cur_input_ids = cur_input_ids[beam_idx, :]
            # 【caption详解】累计整句缓冲必须按相同父行号重排，否则新词会接到错误历史。
            decoding_results = decoding_results[beam_idx, :]
            # 【caption详解】缓存外层通常按注意力层编号索引，module 是键名，不一定是 nn.Module 对象；逐层重排。
            for module in incremental_state:
                # 【caption详解】key 是某层内 prev_key/prev_value 等缓存键；该版本假设参与此循环的值都是4维可重排张量。
                for key in incremental_state[module]:
                    # 【caption详解】index_select(0,indices) 按批量轴重排缓存 [BK,heads,cache_len,head_dim]；须与结果历史完全一致。
                    result = incremental_state[module][key].index_select(0, beam_idx)
                    # 【caption详解】保留前三/后轴，缓存时间轴去掉最后1位，即临时 MASK；下一步改为真实新词重新编码。图片和此前确定文字保留。
                    incremental_state[module][key] = result[:,:,:-1,:]
            
            # 【caption详解】next_ids 为下一轮 MASK 占位 [BK,1]；dtype=long 表示词编号，不是浮点特征。
            next_ids = torch.full(
                (batch_size * self.num_beams, 1), mask_id, dtype=torch.long, device=image.device
            )
            # 【caption详解】新词 [BK] 加单列→[BK,1]，再拼 MASK→[BK,2]；更早历史在缓存，所以输入不用越来越长。
            cur_input_ids = torch.cat([beam_words.unsqueeze(1), next_ids], dim=1)
            # 【caption详解】结果缓冲的当前MASK位置写入选中新词；注意完成候选分支在这行之前已复制了前缀。
            decoding_results[:, cur_len-1] = beam_words
            # update current length
            # 【caption详解】逻辑总长度加1，即已确定文字多了1个，短输入仍两项。
            cur_len = cur_len + 1
            # stop when we are done with each sentence
            # 【caption详解】Python all 检查所有图片都停止，才能整个批量提前退出；单张完成靠PAD占位。
            if all(done):
                break
        
        # select the best hypotheses
        # 【caption详解】tgt=target，tgt_len=[B,1] 保存最终含EOS长度；这里在CPU上建，后面用作切片标量。
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                    dtype=torch.float).fill_(-1e5).to(cur_input_ids.device)
        # 【caption详解】按图片收集最终最优候选，来自已完成容器而非直接取最后活跃beam。
        all_best = []

        # 【caption详解】enumerate 同时给图片索引i与其完成容器 hypotheses；较远处重新提醒 hypothesis 是候选句子。
        for i, hypotheses in enumerate(generated_hyps):
                best = []
                # 【caption详解】hypotheses.hyp 每项(长度归一化分数,token序列)；抽出分数再选优。如果没有完成项会为空，配置/长度逻辑必须保证可完成。
                hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
                _, best_indices = torch.topk(hyp_scores,
                        min(num_keep_best, len(hyp_scores)), largest=True)
                for best_idx, hyp_idx in enumerate(best_indices):
                    # 【caption详解】conf 是长度归一化log分数，不是0到1置信概率；best_hyp 是该候选整数序列。
                    conf, best_hyp = hypotheses.hyp[hyp_idx]
                    best.append(best_hyp)
                    logprobs[i, best_idx] = conf
                    # 【caption详解】在已保存序列后再留1位EOS；如果保存序列含末尾MASK，随后会跳过特殊token，但长度评分仍算了该占位。
                    tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol
                all_best.append(best)
        
        # generate target batch, pad to the same length
        # 【caption详解】decoded 建最终 [B,1,T] 整数结果缓冲，并用 fill_ 原地填PAD。设备与cur_input_ids一致。
        decoded = cur_input_ids.new(batch_size, num_keep_best, self.max_len).fill_(pad_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                # 【caption详解】将最优候选写到结果前缀；batch_idx 是图片号，best_idx 是保留候选号（当前只有0）。
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                # 【caption详解】在保存序列后显式放EOS，后面的剩余位置维持PAD。
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]
        
        # 【caption详解】squeeze(1) 去掉长度1的候选轴得到[B,T]；batch_decode把IDs转成文字，skip_special_tokens去掉CLS/MASK/PAD/SEP等注册特殊词。不是把所有编号直接join。
        captions = self.tokenizer.batch_decode(decoded.squeeze(1), skip_special_tokens=True)
        # 【caption详解】qid 历史名字常指question ID，但本处来自image_id，是图片编号；pred=prediction，生成的描述字符串。
        for qid, pred in zip(image_id, captions):
            self.predictions.append({
                # 【caption详解】单元素张量编号转普通Python数值，方便JSON序列化；image_id=None无法在上面zip遍历，因此实际评估必须提供编号。
                "image_id": qid.item(), 
                # 【caption详解】caption 是JSON结果字段，存单条描述，供COCO等评估器关联image_id。
                "caption": pred, 
            })

    def after_eval(self, **kwargs):
        return self.predictions, "prediction"


def get_handler(args):
    if args.task == "nlvr2":
        return NLVR2Handler()
    elif args.task == "vqav2":
        return VQAHandler()
    elif args.task in ("flickr30k", "coco_retrieval"):
        return RetrievalHandler()
    elif args.task in ("coco_captioning", "nocaps"):
        return CaptioningHandler(args)
    elif args.task in ("imagenet"):
        return ImageNetHandler(args)
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: TaskHandler, epoch: int, start_steps: int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer: Optional[utils.TensorboardLogger] = None, 
        task = None, mixup_fn=None,
):
    # epoch=遍历一轮训练样本；model.train(True) 开启 Dropout 等训练行为，和梯度开关是不同设置。
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # data_loader 每次给一个 batch；任务 Handler 决定 batch 如何进入模型。
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # update_freq 个小 batch 合成一次有效参数更新，故优化步数用整除。
        # data_iter_step=当前微批次序号；update_freq=多少微批次更新一次，// 是整数整除。
        # 例如 update_freq=4，微批次 0/1/2/3 都属于同一个优化 step=0。
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    # lr=learning rate（学习率）；param_group 是优化器管理的一组参数及超参数。
                    # 不同层可有不同 lr_scale，统一调度值乘该比例；不是乘到输入图片上。
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        # 图片、文字 ID、标签都移到目标设备，避免 CPU/GPU 混用报错。
        for tensor_key in data.keys():
            # to 移动当前 batch 字段到设备；non_blocking 请求异步复制，是否真正异步还取决于源内存等条件。
            # 数据字段应为 Tensor；raw 字符串/路径不能直接这样 .to()，所以预处理/collate 必须匹配。
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        # mixup for imagenet finetuning
        if mixup_fn is not None:
            data["image"], data["label"] = mixup_fn(data["image"], data["label"])
        
        if task in ["coco_captioning", "nocaps"]:
            data["global_step"] = global_step

        if loss_scaler is None:
            # **data 将字典键展开为命名参数，由当前任务 Handler 接收。
            results = handler.train_batch(model, **data)
        else:
            # autocast 自动选择部分运算的精度，以减少显存占用。
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        # Handler 返回字典；pop 取走供反传的主损失，其他值可用于日志。
        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            # 累积多个小 batch 的梯度时，先除以次数以保持总梯度尺度。
            loss /= update_freq
            # loss_scaler=None 在此入口表示 DeepSpeed 分支，model 是 engine 包装器，不是任意普通 nn.Module。
            # engine.backward 管反传；不能把这行当成 CPU 上普通 Module 一定具有的接口。
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            # loss_scaler 处理混合精度的缩放/反向/梯度更新；不是一个额外学习目标。
            # clip_grad=max_norm 限制梯度范数；parameters 指定模型参数；update_grad 到累积边界才更新。
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        # 等待 GPU 异步计算完成，保证日志计时一致；这个训练入口显式使用 CUDA，不是开箱即用 CPU 循环。
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value, 
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    # 装饰器使整个评估函数不记录梯度；仍要 model.eval 关闭 Dropout，两个操作都需要。
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()
