# torch=PyTorch，负责 Tensor 运算、设备与求导；nn=neural network，提供有状态的网络/参数对象。
import torch
from torch import nn

# configurable 是项目装饰器，让构造函数可以通过 from_config 将配置转换成显式参数。
from uniperceiver.config import configurable
# 点号表示相邻模块；registry=注册表，用类名查找预测器工厂，不是下载模型的服务。
from .build import PREDICTOR_REGISTRY
# math 是标准数学库（sqrt 开平方）；pickle 是标准对象序列化库，不同于 JSON 文本。
import math
import pickle
# F=functional，PyTorch 函数式接口；如 gelu 激活、normalize 归一化。
import torch.nn.functional as F

# numpy 是数组/数值库，np 是惯用别名；此处 np.log 初始化对数缩放常数，不用于神经网络反传。
import numpy as np

# comm=communication，本地分布式辅助接口：world_size 查进程数，rank 查进程序号。
from uniperceiver.utils import comm
# dist=distributed，跨进程通信；all_gather 收集各进程 Tensor，原调用不自动保留远端求导图。
import torch.distributed as dist
# FP16=16 位浮点；本地 LayerNorm 变体的实际精度行为还要看实现及配置，名字不是所有运算精度保证。
from uniperceiver.modeling.layers import FP16LayerNorm
# AMP=Automatic Mixed Precision；autocast 是上下文管理器，控制块内自动精度选择。
from torch.cuda.amp import autocast



__all__ = ["EmbedClsAsRetrievalPredictor"]
def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

@PREDICTOR_REGISTRY.register()
# Embed=embedding（嵌入），Cls=classification（分类），AsRetrieval=把分类表示成候选检索。
# Predictor=预测器；继承 nn.Module 后参数会参加 state_dict、设备转移及优化器收集。
class EmbedClsAsRetrievalPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        temperature,
        use_norm,
        temp_learn,
        mb_list,
        queue_len,
        feat_dim,
        task2tempname,
        fc_prompt_feature_index,
        output_proj,
        cfg,

    ):
        super(EmbedClsAsRetrievalPredictor, self).__init__()
        self.cfg = cfg
        self.use_norm = use_norm
        self.temp_learn = temp_learn
        # temp=temperature（温度），learn=可学习；logit_scale 保存 log(1/τ)，不是 τ 本身。
        # Parameter 注册可学习标量；torch.ones([]) 是零维标量 Tensor，不是长度为 0 的向量。
        if temp_learn:
            self.logit_scale_img_cls = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_video_cls = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_text_mlm = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_text_caption = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_caption = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_mlm = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_retrieve = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_tqa_mlm = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_tqa_caption = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_tqa_retrieve = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
            self.logit_scale_downstream = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        else:
            # 固定温度分支显式 .cuda()，不能假设构造时兼容纯 CPU；这些普通 Tensor 也不是注册 buffer。
            self.logit_scale_img_cls = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_video_cls = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_text_mlm = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_text_caption = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_caption = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_mlm = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_retrieve = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_tqa_mlm = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_tqa_caption = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_tqa_retrieve = torch.ones([]).cuda() * np.log(1 / temperature)
            self.logit_scale_downstream = torch.ones([]).cuda() * np.log(1 / temperature)


        # 2 读作 to：任务名 → 温度属性名。例如不同任务选不同 logit_scale_*。
        self.task2tempname = task2tempname


        self.memory_save = []
        self.queue_len = queue_len
        self.feat_dim = feat_dim
        self.fc_prompt_feature_index = fc_prompt_feature_index
        # mb=memory bank（记忆库）；queue=先进先出特征缓存，保存历史候选以增加负例。
        # feat_dim=feature dimension（向量宽度 D）；queue_len=缓存条数 K。
        for task_name in mb_list:
            self.memory_save.append(task_name)
            # register_buffer 保存随模型移动/存盘但不由优化器学习的 Tensor。h1/h2 分别缓存两侧特征。
            self.register_buffer('queue_h1_{}'.format(task_name), torch.randn(queue_len, feat_dim ))
            self.register_buffer('queue_h2_{}'.format(task_name), torch.randn(queue_len, feat_dim ))
            setattr(self, 'queue_h1_{}'.format(task_name), nn.functional.normalize(getattr(self, 'queue_h1_{}'.format(task_name)), dim=1))
            setattr(self, 'queue_h2_{}'.format(task_name), nn.functional.normalize(getattr(self, 'queue_h2_{}'.format(task_name)), dim=1))

            self.register_buffer("queue_ptr1_{}".format(task_name), torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_ptr2_{}".format(task_name), torch.zeros(1, dtype=torch.long))

        pass

        self.output_proj = output_proj
        if self.output_proj:
            # if cfg.MODEL.LN_FP32:
            #     self.ln_post = CustomLayernorm(feat_dim)
            # else:
            #     self.ln_post = nn.LayerNorm(feat_dim)
            if self.cfg.SOLVER.FORCE_LN_FP16:
                self.ln_post = FP16LayerNorm(feat_dim)
            else:
                self.ln_post = nn.LayerNorm(feat_dim)
            self.proj = nn.Linear(feat_dim, feat_dim)

        if cfg.MODEL.FEATURE_GATHER_FORCE:
            assert cfg.DATALOADER.STRATEGY == 'turn'
            self.gather_feature = True
        else:
            self.gather_feature = len(cfg.TASKS) == 1 and getattr(cfg.MODEL, "FEATURE_GATHER", False)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, q1, q2, task_name):
        """Update queue."""
        # gather keys before updating queue


        batch_size1 = q1.shape[0]
        batch_size2 = q2.shape[0]

        ptr1 = int(getattr(self, "queue_ptr1_{}".format(task_name)))
        ptr2 = int(getattr(self, "queue_ptr2_{}".format(task_name)))

        assert self.queue_len % batch_size1 == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)

        getattr(self, 'queue_h1_{}'.format(task_name))[ptr1:ptr1+batch_size1, :] = q1 # save text features
        getattr(self, 'queue_h2_{}'.format(task_name))[ptr2:ptr2+batch_size2, :] = q2 # save img features

        ptr1 = (ptr1 + batch_size1) % self.queue_len  # move pointer
        ptr2 = (ptr2 + batch_size2) % self.queue_len  # move pointer


        getattr(self, "queue_ptr1_{}".format(task_name))[0] = ptr1
        getattr(self, "queue_ptr2_{}".format(task_name))[0] = ptr2

        pass

    def replace_weight(self, weight):
        pass

    def replace_module_hidden(self,dense, layer_norm):
        pass

    @classmethod
    def from_config(cls, cfg):

        mb_list = []
        task2tempname = {}
        if len(cfg.TASKS) > 0:
            for task_config in cfg.TASKS:
                if 'MODEL' in task_config and task_config['MODEL'].get('MEMORY_BANK', False):
                    mb_list.append(task_config['NAME'])
                task2tempname[task_config['NAME']] = task_config['MODEL']['TEMP_NAME']

        ret = { "temperature": cfg.MODEL.PRED_TEMPERATURE,
               "use_norm": cfg.MODEL.PRED_USE_NORM,
               'temp_learn': getattr(cfg.MODEL, "LEARN_TEMP", False),
               'mb_list': mb_list,
               'queue_len': cfg.MODEL.QUEUE_LEN,
               'feat_dim': cfg.MODEL.ENCODER_DIM,
               'task2tempname': task2tempname,
               "fc_prompt_feature_index": cfg.MODEL.FC_PROMPT_INDEX,
               "output_proj": cfg.MODEL.OUTPUT_PROJ,
               "cfg": cfg,
              }
        print(f'********* using temperature {cfg.MODEL.PRED_TEMPERATURE} **********')

        return ret

    @classmethod
    def add_config(cls, cfg):
        pass

    def test_forward(self, logits):
        return { "output": logits }

    def postproj(self, hidden_states):
        x = self.ln_post(hidden_states)
        if self.proj is not None:
            x = self.proj(x)
        return x



    def forward(self,
                input_sample_list,
                target_sample_list,
                shared_target_sets,
                target_set_list,
                target_idx_list,
                task_info,
                **kwargs):

        # 候选可以随当前样本提供，也可以从共享候选集合里读取。
        if len(target_sample_list) > 0:
            q2_hidden_states = target_sample_list[0]['data']
        else:
            if len(target_set_list) > 1:
                raise NotImplementedError('only one target supported now')
            target_set_name = target_set_list[0]
            q2_hidden_states = shared_target_sets[target_set_name][0]['data']


        # q1 是待判断的输入，q2 是候选目标；两边稍后形成匹配分数。
        q1_hidden_states = input_sample_list[0]['data']

        # 取候选序列的第 0 个汇总 token，得到每个候选的整体表示。
        # hidden=隐表示，states=状态；q2 此时 [C,Ly,D]，取 [:,0] 变为 [C,D]。
        # C 是候选数，Ly 是候选序列长度；q1/q2 是匹配的两侧，不是注意力的 Q/K/V。
        q2_hidden_states = q2_hidden_states[:, 0]


        task_type = task_info.get('task_type')

        if task_type in ['image_classification', 'video_classification']:
            # q1 原为 [B,Lx,D]；以第 0 个汇总 token 表示整个输入，得到 [B,D]。
            q1_hidden_states = q1_hidden_states[:, 0]
        elif task_type in ['image_retrieval', 'video_retrieval']:
            # q1 原为 [B,Lx,D]；以第 0 个汇总 token 表示整个输入，得到 [B,D]。
            q1_hidden_states = q1_hidden_states[:, 0]
        elif task_type == 'text_mlm':
            # MLM 只预测被遮位置；target_idx=-1 的位置不计入目标。
            # ne=not equal，逐位置比较得到 bool mask [B,T]；-1 是忽略位置，不是词表编号。
            mask_tokens = target_idx_list[0].ne(-1) # -1 is unmasked position
            q1_hidden_states = q1_hidden_states[:, -mask_tokens.size(1):][mask_tokens]
            target_idx_list[0] = target_idx_list[0][mask_tokens]
        elif task_type in ['image_caption', 'video_caption']:
            # 训练时有多个目标词；推理时只需最后位置来预测下一个词。
            if self.training:
                sample_info = input_sample_list[0]['sample_info']
                if isinstance(sample_info, list):
                    sample_info = sample_info[0]
                text_length = sample_info['data_length'][-1] // 2
                q1_hidden_states = q1_hidden_states[:, -text_length:, :]
                # ne=not equal，逐位置比较得到 bool mask [B,T]；-1 是忽略位置，不是词表编号。
                mask_tokens = target_idx_list[0].ne(-1) # -1 is padding position
                q1_hidden_states = q1_hidden_states[mask_tokens] # .flatten(0, 1)
                target_idx_list[0] = target_idx_list[0][mask_tokens] # .flatten(0, 1)
            else:
                q1_hidden_states = q1_hidden_states[:, -1]
        elif task_type in ['text_classification', 'vqa']:
            sample_info = input_sample_list[0]['sample_info']
            if isinstance(sample_info, list):
                sample_info = sample_info[0]

            sample_infos = sample_info if isinstance(sample_info, list) else sample_info['sample_info_per_sample'][-1]
            if 'spe_index' in sample_infos[0]:
                text_length = sample_info['data_length'][-1]
                q1_hidden_states = q1_hidden_states[:, -text_length:, :] # get text part; remove the first spe or the prompt embedding part
                # gather spe representation from the 'spe_index' from text part via index of spe token
                # 为每个样本建立要取的特殊 token 下标，并扩展到全部特征维。
                spe_index = torch.tensor([si['spe_index'] for si in sample_infos], device=q1_hidden_states.device).view(-1, 1, 1).expand(-1, -1, q1_hidden_states.size(2))
                # gather 从序列中精准取指定 token，而非固定取第 0 位。
                q1_hidden_states = torch.gather(q1_hidden_states, 1, spe_index)[:, 0]
            else:
                # q1 原为 [B,Lx,D]；以第 0 个汇总 token 表示整个输入，得到 [B,D]。
                q1_hidden_states = q1_hidden_states[:, 0]

        else:
            raise NotImplementedError


        # 可选投影把输入/候选表示送到便于比较的同一空间。
        if self.output_proj:

            q1_hidden_states = self.postproj(q1_hidden_states)
            q2_hidden_states = self.postproj(q2_hidden_states)

        # feat=feature，保存预测器使用的输入向量，供可选 fc prompt 分支使用；没有 detach。
        feat = q1_hidden_states


        if len(target_sample_list) ==  0:
            # in1k
            logits = self._forward(q1_hidden_states, q2_hidden_states, task_name=task_info.get("task_name", None))


            ret = { "logits": [logits],  "feats": [feat],  "loss_names": [''] }
            if len(target_idx_list) > 0:
                ret.update({"targets": [target_idx_list[0]]})

            if not self.training:
                ret_test = self.test_forward(logits)
                ret.update(ret_test)
                # ret = self.test_forward(logits)



        else:
            # image and text retrieval in one forwarding:

            if not self.training:
                return {
                    "input_feats": q1_hidden_states / q1_hidden_states.norm(dim=-1, keepdim=True),
                    "tgt_feats": q2_hidden_states / q2_hidden_states.norm(dim=-1, keepdim=True),
                }

            # 跨 GPU 汇总图文向量，增加对比学习中的负例数。
            if self.gather_feature:
                local_q1 = q1_hidden_states
                local_q2 = q2_hidden_states
                packed_feature = torch.cat([local_q1, local_q2], dim=1).float()

                gathered_features = [ torch.zeros_like(packed_feature) for _ in range(comm.get_world_size())]

                # gathered_features 是各进程接收缓冲列表；当前调用收集数值，远端特征不带原求导图。
                dist.all_gather(gathered_features, packed_feature)

                # 使用本地 packed_feature 替换本进程收集副本，以保留本地梯度；本地批次排列在最前。
                # 两侧特征按相同顺序收集，所以之后 arange 配对仍能成立。
                all_features = torch.cat([packed_feature] +
                                         gathered_features[:comm.get_rank()] +
                                         gathered_features[comm.get_rank() + 1:]).to(local_q1)

                q1_hidden_states, q2_hidden_states = torch.split(all_features, [local_q1.size(1), local_q2.size(1)], dim=1)


            if task_info.get("task_name", None) in self.memory_save:
                # retrieval task with memory buffer
                logits, logits_per_cls = self._forward_with_mb(
                    q1_hidden_states,
                    q2_hidden_states,
                    task_name=task_info.get("task_name", None))
            else:
                logits, logits_per_cls = self._forward(q1_hidden_states,
                                                       q2_hidden_states,
                                                       task_name=task_info.get(
                                                           "task_name", None),
                                                       mutual_retrieval=True)

            # 当前 batch 的第 i 个图像与第 i 段文字对应，故标签是 0..B-1。
            target = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
            target_per_cls = target

            ret = {
                "logits": [logits, logits_per_cls],
                "targets": [target, target_per_cls],
                "loss_names": ['i2t', 't2i'],
            }

        return ret


    def _forward(self, g, cls_name, task_name, mutual_retrieval=False,):
        # 此处变量虽叫 temperature，实际是 exp(log(1/τ))=1/τ，即分数乘数。
        temperature = self.temperature_task(task_name)
        if self.cfg.SOLVER.FORCE_TEMP_FP16:
            temperature = temperature.half()
        if self.temp_learn and temperature > 100.0:
            temperature = 100.0

            # getattr(self, self.task2tempname[task_name]).data.clamp_(max=math.log(100.0))

        # norm=范数；将每行特征除以自身长度，点积就成为余弦相似度。
        # 原实现没有 eps 防零；极端零向量可能产生非有限值，不能宣称总是数值安全。
        if self.use_norm:
            if not self.cfg.SOLVER.FORCE_NORM_FP16:
                g = g / g.norm(dim=-1, keepdim=True)
                cls_name = cls_name / cls_name.norm(dim=-1, keepdim=True)
            else:
                with autocast(enabled=False):
                    g = g / g.norm(dim=-1, keepdim=True)
                    cls_name = cls_name / cls_name.norm(dim=-1, keepdim=True)


        # g 是输入特征 [B,D]，cls_name 虽叫类别名，实际已经是候选向量 [C,D]。
        # .t() 对二维张量转置到 [D,C]；@ 矩阵乘法 → [B,C]，再乘 1/τ。
        # logits 是未归一化分数，后续 CrossEntropyLoss 内部处理 log-softmax；这里不返回概率。
        logits = (g @ cls_name.t()) * temperature

        if mutual_retrieval:
            # mutual=双向，per_cls=按候选一侧查询；转置将图找文分数变成文找图 [C,B]。
            logits_per_cls =  logits.transpose(0, 1)
            return logits, logits_per_cls
        return logits

    def _forward_with_mb(self, g, cls_name, task_name):
        # 此处变量虽叫 temperature，实际是 exp(log(1/τ))=1/τ，即分数乘数。
        temperature = self.temperature_task(task_name)
        if self.temp_learn and temperature > 100.0:
            temperature = 100.0

        # if self.temp_learn:
        #     getattr(self, self.task2tempname[task_name]).data.clamp_(max=math.log(100.0))
        # norm=范数；将每行特征除以自身长度，点积就成为余弦相似度。
        # 原实现没有 eps 防零；极端零向量可能产生非有限值，不能宣称总是数值安全。
        if self.use_norm:
            g = g / g.norm(dim=-1, keepdim=True)
            cls_name = cls_name / cls_name.norm(dim=-1, keepdim=True)

        logits_per_image = (g @ cls_name.t()) * temperature

        logits_per_cls =  logits_per_image.transpose(0, 1)

        logits_per_image_neg = (g @ getattr(self, 'queue_h1_{}'.format(task_name)).clone().detach().t()) * temperature

        logits_per_cls_neg = (cls_name @ getattr(self, 'queue_h2_{}'.format(task_name)).clone().detach().t()) * temperature

        self._dequeue_and_enqueue(cls_name, g, task_name) # reverse sequnce to save

        return torch.cat([logits_per_image, logits_per_image_neg], dim=-1) , torch.cat([logits_per_cls, logits_per_cls_neg], dim=-1)

    @property
    def temperature_dict(self):
        return {
            'temperature/img_cls': 1/self.logit_scale_img_cls.exp(),
            'temperature/video_cls': 1/self.logit_scale_video_cls.exp(),
            'temperature/text_mlm': 1/self.logit_scale_text_mlm.exp(),
            'temperature/text_caption': 1/self.logit_scale_text_caption.exp(),
            'temperature/caption': 1/self.logit_scale_caption.exp(),
            'temperature/mlm': 1/self.logit_scale_mlm.exp(),
            'temperature/retrieve': 1/self.logit_scale_retrieve.exp(),
            'temperature/tqa_mlm': 1/self.logit_scale_tqa_mlm.exp(),
            'temperature/tqa_caption': 1/self.logit_scale_tqa_caption.exp(),
            'temperature/tqa_retrieve': 1/self.logit_scale_tqa_retrieve.exp(),
            'temperature/downstream': 1/self.logit_scale_downstream.exp(),
        }

    def temperature_task(self, taskname):
        # getattr 按字符串取得对象属性；映射选定可学习 logit_scale，再 exp 恢复正乘数。
        # 同文件 temperature_dict 用 1/exp 恢复真正 τ，两者要区分。
        return getattr(self, self.task2tempname[taskname]).exp()
