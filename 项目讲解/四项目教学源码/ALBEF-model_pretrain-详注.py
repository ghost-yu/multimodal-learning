# 教学版：原文件 ALBEF/models/model_pretrain.py。只增加注释，保留原可执行结构。
# B=当前卡样本数，P=图像 token 数，T=文字 token 数，E=对比空间维度，Q=队列长度。
# _m=momentum teacher 动量教师；feat=整图/整句向量，embeds=每个位置的向量序列。
# import：partial 预填 LayerNorm 的 eps；torch 张量与分布式；F 是 normalize/softmax/交叉熵函数。
# nn 为 Module/Linear/Parameter；VisionTransformer 与 BertForMaskedLM 是本项目网络部件。
'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

# functools 是 Python 标准库；partial(函数, 已指定参数) 返回一个新可调用对象。
# 这里 partial(nn.LayerNorm, eps=1e-6) 让 ViT 后续建每个 LayerNorm 时自动带 eps。
from functools import partial
# models.vit 是本仓库模块：VisionTransformer 把图片变成 patch token；
# interpolate_pos_embed 对 DeiT 权重的位置向量做插值，适配这里的图片尺寸/patch 网格。
from models.vit import VisionTransformer, interpolate_pos_embed
# models.xbert 是本仓库改造的 BERT，不应直接等同未经修改的 transformers.BertForMaskedLM。
# BertConfig 保存层数/隐藏维等配置；BertForMaskedLM 支持本文的 text/fusion 与遮词预测。
from models.xbert import BertConfig, BertForMaskedLM

# torch 管张量、无梯度上下文、随机采样及多进程通信；下面所有张量维度均按 batch 在前。
import torch
# F 是 torch.nn.functional：normalize 做 L2 单位化，softmax/log_softmax 变概率，cross_entropy 算分类损失。
import torch.nn.functional as F
# nn 是 torch.nn：Module 是模型父类，Linear 是带权线性层，Parameter 注册可训练标量。
from torch import nn

# 本文件保留原仓库的 numpy/random 导入；当前预训练 forward 的采样使用 torch 而非这两个名字。
import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        # text_encoder 是已有 BERT 权重名/路径；tokenizer 把文字变成 ID；config 是超参数字典。
        # init_deit=True 会尝试下载 DeiT 视觉初始化权重。形参 temp=0.07 在此实现未实际用于
        # self.temp 初始化；真正使用 config['temp']，读代码时不要误以为改 temp 形参有效。
        super().__init__()
        
        # 保存分词器供 mask() 读取 PAD、CLS、MASK 的整数 ID；此处不调用 tokenizer。
        self.tokenizer = tokenizer 
        # mlm_probability 是每个可遮盖文字位置被选中的概率，例如 0.15。
        self.mlm_probability = config['mlm_probability']
        # embed_dim=E，图像和文字通过各自投影后共享的对比学习向量宽度。
        embed_dim = config['embed_dim']
     
        # 视觉编码器输出每个图像 patch 的序列，不只是一个分类结果。
        self.visual_encoder = VisionTransformer(
            # img_size 是输入图片边长；patch_size=16 表示每个 16×16 小块一个 token。
            # embed_dim=768 是 ViT 内部宽度，depth=12 是层数，num_heads=12 是每层头数。
            # mlp_ratio=4 指 FFN 隐藏宽度约为 4×768；qkv_bias 允许 QKV 投影带偏置。
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            # torch.hub.load_state_dict_from_url 下载/缓存权重并读为字典；check_hash 验证文件摘要。
            # map_location='cpu' 先在 CPU 打开，避免检查点来源设备与本机不一致。
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            # checkpoint 是外层字典；state_dict 才是“层参数名→张量”。
            # pos_embed 为位置编码，图片网格变化会使其长度不同，因此先插值。
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            # strict=False 容许 DeiT 分类头等多余键与本模型缺失键；msg 记录哪些键未匹配。
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        # vision_width 是 ViT 产出的每 token 宽度 Dv，应与上面 ViT 配置相容。
        vision_width = config['vision_width']      
        # 从 JSON 文件读取 BERT 层数、词表大小等结构参数；不会单独下载权重。
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        # from_pretrained 读取 text_encoder 指定的预训练权重，并按 bert_config 建文字网络。
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        # text_width=Dt，从实际创建的 BERT 配置取隐藏维，避免凭空假定与视觉 768 相同。
        text_width = self.text_encoder.config.hidden_size
        # 两个投影层把图像/文本 CLS 都映射到同一 embed_dim，才能做相似度矩阵。
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        # 对比损失的温度参数：相似度除以 temp 后进入 softmax。
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        # torch.ones([]) 是零维标量张量；包为 Parameter 后优化器会训练温度。
        # Q 是历史候选队列长度；momentum 控制教师参数向学生靠近的速度。
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        # ITM 头把融合后的文字 CLS [Dt] 映射到 [2]：不匹配/匹配两类原始分数。
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        # 四个 _m 模型都是教师副本：结构对应学生，初始化后不走常规梯度更新。
        
        # model_pairs 每项都是 [学生子网络,对应教师子网络]，供复制参数和动量更新循环使用。
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()
        # copy_params 第一次把学生参数复制到教师，保证二者初始预测相同。

        # create the queue
        # register_buffer 会保存/迁移队列但不让优化器训练它；每一列是一张历史图像的 E 维特征。
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        # text_queue 同为 [E,Q]；queue_ptr 长度 1，保存下一次写入的起始列。
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        # dim=0 意味对每一列 E 维特征归一化。两个 buffer 的新张量重新登记到同名属性。
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)



    def forward(self, image, text, alpha=0):
        # image:[B,3,H,W]；text 是 tokenizer 的批量编码对象，含 input_ids:[B,T]、attention_mask:[B,T]。
        # alpha 混合教师软目标与真实硬标签；三个损失分别监督补词、全局对齐、图文真假。
        # 本函数依次计算：图文对齐 ITA、图文匹配 ITM、掩码词预测 MLM。
        with torch.no_grad():
            # clamp_ 原地限制温度在 [0.001,0.5]，避免除以过小/过大温度导致数值不稳定。
            self.temp.clamp_(0.001,0.5)
        
        # 学生 ViT 输出每个图像 token 的向量 [B,P,Dv]；第 0 个通常是 CLS 汇总 token。
        image_embeds = self.visual_encoder(image) 
        # size()[:-1] 去掉特征维，得到 [B,P]；ones 说明这些图像 token 全部有效。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        # 取第 0 个 CLS token 表示整张图，再投影并归一化成 [B,E]。
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)

        # text_encoder.bert 只用 BERT 编码模块；mode='text' 此刻不读图像。
        # attention_mask 标 PAD；return_dict=True 让返回值能用 .last_hidden_state 属性读取。
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        # text_embeds 是每个词的上下文向量 [B,T,Dt]，供稍后的图文融合与难负例复用。
        text_embeds = text_output.last_hidden_state
        # 文本也取 CLS 得到 [B,E]；此时还处于 mode='text'，未与图像融合。
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
             
        # get momentum features
        with torch.no_grad():
            # 带 _m 的教师网络不经梯度更新，而是对学生参数做滑动平均。
            self._momentum_update()
            # 教师视觉网络输入同一批 image，但参数是缓慢滑动平均值；输出形状同学生 [B,P,Dv]。
            image_embeds_m = self.visual_encoder_m(image) 
            # 教师也把图像 CLS 映射到 [B,E]，但此处不计算梯度。
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)
            # 当前 batch 的教师特征 + 历史队列：扩大候选集合。
            # 形状 [E,B+Q]，用于与学生的 [B,E] 相乘。
            # 当前教师图像 B 列 + 历史队列 Q 列 → [E,B+Q]；t 转置，detach 不追溯旧梯度。
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)
            # 教师文字同样按 mode='text' 处理原始 token ID；先不做图文交叉注意力。
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            # 文本候选同样由当前 batch 与历史队列拼成 [E,B+Q]。
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            # [B,E] @ [E,B+Q] -> [B,B+Q]，每行是一张图对所有文本的分数。
            # i2t=image-to-text；[B,E]@[E,B+Q] 给出每张图对 B+Q 段文字的分数。
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            # t2i=text-to-image：每段文字对所有 B+Q 个图像候选的教师分数，形状同样 [B,B+Q]。
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            # 建全零硬标签矩阵，行是当前 B 个查询、列是 B+Q 个候选；设备与 image 一致。
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            # 当前 batch 的第 i 张图配第 i 段文字；队列列没有一热正例。
            # 本 batch 的第 i 图与第 i 文是一对；原地把前 B 列的对角线改 1。
            sim_targets.fill_diagonal_(1)

            # alpha 混合教师的软概率与真实配对的一热标签，得到蒸馏目标。
            # alpha 在训练入口传来；混合教师概率与真实一热标签。dim=1 表示每行候选之和为 1。
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # t2i 也混教师分布与同一批配对的对角硬标签，返回每行和为 1 的目标概率。
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # 学生预测要接近上面教师与真实标签混合出的目标分布。
        sim_i2t = image_feat @ text_feat_all / self.temp
        # 与 *_m 不同，这两行的左侧查询来自学生网络，梯度要沿 image_feat/text_feat 返回。
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        # 软标签交叉熵：逐候选求 target × log(预测概率)，再对 batch 平均。
        # 软目标交叉熵：每候选 target×log概率，逐行求和取负，再对 B 张图平均。
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        # 另一方向同理；mean() 后各是一个标量张量，loss_ita 是两个方向的平均。
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        # 两个方向平均，既训练“图找文”，也训练“文找图”。
        loss_ita = (loss_i2t+loss_t2i)/2

        # 当前教师特征进入队列，供后续 batch 充当额外对比候选。
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        # 队列保存的是教师的整图/整句 E 维向量，不是像素或逐 token 表示。

        ###=================================###
        # forward the positve image-text pair
        # mode='fusion' 让文字 token 通过交叉注意力读取图像 patch。
        # 正例仍是数据集原始配对。encoder_embeds=text_embeds 是前面学生纯文字编码的 [B,T,Dt]；
        # encoder_hidden_states=image_embeds 是同一批学生图像的 [B,P,Dv]。
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,
                                        # 文字 attention_mask 与原 text.input_ids 对应，遮掉 PAD 位置。
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            # bs 即 batch size，本进程这批有多少对图文；不是队列 Q 的长度。
            bs = image.size(0)          
            # 只在当前 batch 抽难负例；队列中只有向量，缺少融合所需 token。
            # [:,:bs] 只保留当前批 B 列；历史队列没有逐 token 表示，无法送进 fusion。
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            # weights_t2i 的第 b 行：第 b 段文字对当前 B 张图像的采样概率。
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            # 采负例时把正确配对的对角线概率清零，避免抽到真配对。
            # 采难负例前排除正确图文；否则可能把真实配对当成负例。
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        # *_neg 表示按相似度选择的错配样本；此列表先收每段文字配的错误图像序列。
        image_embeds_neg = []    
        for b in range(bs):
            # b 是当前文字的行号；weights_t2i[b] 给这段文字对 B 张图的分布。
            # 根据相似度概率抽错图，越像正确图片越可能被抽到，因而更难。
            # multinomial 按概率抽一个错误图片下标；相似度越高被抽中概率越大。item 取 Python 整数。
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            # image_embeds[neg_idx] 是某张错误图像的全部 P 个 token，不是一个 E 维全局向量。
            image_embeds_neg.append(image_embeds[neg_idx])
        # stack 把 B 个 [P,Dv] 排成 [B,P,Dv]；与原 image_embeds 形状一致。
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        # 错误文字的 attention_mask 也必须成对收集，否则融合时会把 PAD 当真实词。
        text_atts_neg = []
        for b in range(bs):
            # 对每张图也抽一段相似但不匹配的错误文字。
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        # 两个列表分别 stack 成 [B,T,Dt] 与 [B,T]，顺序一一对应。
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        # dim=0 拼的是样本批量：前 B 个原文字，后 B 个错误文字，合计 2B 个。
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        # 对应文字 mask 保持同样顺序，避免文字与 mask 错位。
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        # 前 B 个错误图片配原文字，后 B 个原图片配错误文字；与 text_embeds_all 顺序必须对应。
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        # 两组图像 token 都完全有效，所以把同一个 image_atts 复制两份。
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        # 前 B 对=(原文,错图)，后 B 对=(错文,原图)；这是 2B 个融合负例。
        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        # 正例和两组负例的融合 CLS 拼成 [3B,D]，统一送入二分类头。
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        # vl=vision-language；[3B,Dt]→线性头 [3B,2]，每行两个未归一化分类分数。
        vl_output = self.itm_head(vl_embeddings)            

        # 标签顺序：[B 个真实=1, 2B 个错配=0]，总共 3B 个二分类样本。
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        # 真配对 B 个、两类难负配对共 2B 个，训练二分类匹配头。
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        # F.cross_entropy 内部做 log_softmax + 取正确类负对数，输出一个标量损失。
        
        ##================= MLM ========================##                
        # clone 保留 text 对象的原始词 ID；下方 mask 会原地替换被选中的位置。
        input_ids = text.input_ids.clone()
        # labels 留住遮盖前正确词 ID；本函数稍后把未选中位置改 -100，不应与输入共用同一张量。
        labels = input_ids.clone()

        # full(shape,p) 创建每个位置概率都是 p 的 [B,T] 矩阵；还没有真正掷硬币。
        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        # 随机遮住一部分文字；labels 保留原词，其余位置设为忽略值。
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            # 教师给被遮位置产生词表软概率，供学生蒸馏。
            # 教师阅读已被 mask 的 input_ids 与教师图片 image_embeds_m；return_logits=True 直接取词表分数。
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       # 学生读同一份遮盖文字，但看学生图像 image_embeds；labels 是正确词。
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       # 教师原始 logits 沿最后的词表维 softmax，作为每个位置的软词分布。
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss        
        # loss_mlm 的硬标签/软标签具体混法在本仓库 models/xbert.py 的 BertForMaskedLM.forward 内。

        # 返回顺序与实际计算顺序不同；训练脚本按此顺序解包并求和。
        return loss_mlm, loss_ita, loss_itm

        

    @torch.no_grad()    
    def copy_params(self):
        # @torch.no_grad 让参数复制不记录计算图；此函数仅初始化一次教师权重。
        for model_pair in self.model_pairs:           
            # zip 并行迭代学生/教师的对应参数；要求两边网络结构和参数顺序相同。
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # .data.copy_ 把学生参数值复制给教师；末尾 _ 表示原地写入。
                param_m.data.copy_(param.data)  # initialize
                # 教师此后不由 optimizer 的 backward 梯度更新，而由下面 _momentum_update 更新。
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        # 每次 forward 在教师抽特征前执行。self.momentum 来自 config，不是学习率。
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # 新教师 = m×旧教师 + (1-m)×学生；m 越接近 1，教师变化越慢。
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # image_feat/text_feat 是当前本进程教师的 [B,E] 全局向量；队列本体为 [E,Q]。
        # gather keys before updating queue
        # concat_all_gather 将所有进程的 batch 沿第 0 维拼接，得到全局 batch 候选。
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        # queue_ptr 是形如 [ptr] 的整数 buffer；int(...) 提出当前写列号。
        ptr = int(self.queue_ptr)
        # batch_size 已在上面经过 all_gather，可能是所有 GPU 的总数；队列大小必须是它的整数倍。
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # image_feats.T 从 [全局B,E] 转 [E,全局B]，才能写进队列的连续列。
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        # 对长度为 1 的 queue_ptr 原地更新；下次进入本函数会从新位置写入。
        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        # input_ids:[B,T] 是待替换词 ID，vocab_size 是词表数，device 是随机词要去的设备。
        # targets 可选，存正确词；masked_indices 可选，显式告诉哪些位置被选；
        # probability_matrix 可选，给每位置被选概率。返回修改后的 input_ids（及可选 targets）。
        if masked_indices is None:                                       
            # bernoulli(p) 每格按概率 p 抽 0/1，再转 bool；得到 [B,T] 的遮盖位置矩阵。
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        # PAD 与句首 CLS 都不是正常预测目标，强制对应位置为 False。
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            # ~ 是逐元素布尔取反；非遮盖位置标签设 -100，交叉熵按 ignore_index 忽略。
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        # 用布尔索引原地替换这些被选中且进入 80% 分支的词 ID。
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # 只从未被 80% [MASK] 分支选中的遮盖位置再抽一半：0.2×0.5=0.1 换随机词。
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        # randint 上界不含 vocab_size，得到 [0,vocab_size) 的随机 ID；只在 indices_random 为真处写入。
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        # get_world_size 返回进程数；每个进程都预留一份同形状接收槽。
        for _ in range(torch.distributed.get_world_size())]
    # all_gather 将每张卡的 tensor 拷贝到所有进程的 tensors_gather；async_op=False 等待完成。
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # 按 batch 维接起来；如果各卡 [B,E]，返回 [进程数×B,E]。此操作不传回梯度。
    output = torch.cat(tensors_gather, dim=0)
    return output

