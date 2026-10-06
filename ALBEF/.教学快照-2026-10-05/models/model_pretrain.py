'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

# partial 预先固定 LayerNorm 的 eps 参数；构造 ViT 的每层归一化时复用。
from functools import partial
# 本项目 ViT 图像编码器；interpolate_pos_embed 在图片分辨率改变时调整位置向量。
from models.vit import VisionTransformer, interpolate_pos_embed
# xbert.py 是本项目改造的 BERT，支持 mode='text' / 'fusion' 与软标签 MLM。
from models.xbert import BertConfig, BertForMaskedLM

# torch 是张量主库；F 是无参数函数式算子；nn 包含可训练层和 Module 基类。
import torch
import torch.nn.functional as F
from torch import nn

# np 和 random 是原仓库保留的导入；本文件主 forward 没有使用它们。
import numpy as np
import random


class ALBEF(nn.Module):
    # ALBEF = ALign BEfore Fuse：先用全局图文向量对齐，再用交叉注意力融合。
    # 继承 nn.Module 后，self 中登记的层会被优化器发现，model(...) 会进入 forward。
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        # text_encoder 是预训练 BERT 名称/路径；tokenizer 提供 PAD/CLS/MASK 的整数 ID。
        # config 是 YAML 读取后的字典；temp 默认参数未单独使用，实际取 config['temp']。
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        # embed_dim=对比空间维数 E（配置 256），与 BERT/ViT 内部宽度 768 不同。
        embed_dim = config['embed_dim']
     
        # 视觉编码器输出每个图像 patch 的序列，不只是一个分类结果。
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        # init_deit=True 从网络获取 DeiT 的视觉权重；不是训练时每步下载。
        if init_deit:
            # checkpoint 是下载的参数字典；先在 CPU 读，随后装入当前视觉编码器。
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            # 原权重常对应不同图片分辨率；插值 patch 位置表以匹配当前 image_res。
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        # vision_width 是每个视觉 token 的内部宽度 Dv；bert_config 保存文字网络结构。
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        # text_width 是每个文字 token 的内部宽度 Dt；与 E 不是同一维。
        text_width = self.text_encoder.config.hidden_size
        # 两个投影层把图像/文本 CLS 都映射到同一 embed_dim，才能做相似度矩阵。
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        # 对比损失的温度参数：相似度除以 temp 后进入 softmax。
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        # queue_size=Q 是历史特征列数；momentum 是教师参数的滑动平均系数。
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        # 每个二元列表依次是 [学生模块,教师模块]；zip 后逐参数复制/滑动更新。
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        # 教师初值与学生一致，并关闭教师参数的梯度更新。
        self.copy_params()

        # create the queue
        # buffer 会随模型保存/搬设备，但优化器不会把它当可训练参数。
        # 队列形状 [E,Q]：每列是一条旧图片/文字的全局向量，不是原始像素/词。
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        # dim=0 表示把队列每一列各自归一化成单位向量，便于余弦相似度比较。
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)



    def forward(self, image, text, alpha=0):
        # 本函数依次计算：图文对齐 ITA、图文匹配 ITM、掩码词预测 MLM。
        # image 是 [B,3,H,W] 浮点图片；text 是含 input_ids/attention_mask 的编码对象。
        # alpha 是硬标签与动量教师软目标的混合权重，0 表示只用硬标签。
        with torch.no_grad():
            # clamp_ 末尾 _ 表示原地修改；限制可训练温度避免分母接近 0。
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        # image_embeds [B,P,Dv]；去掉最后 Dv 后造 [B,P] 全 1 的有效位置标记。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        # 取第 0 个 CLS token 表示整张图，再投影并归一化成 [B,E]。
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)

        # mode='text' 只跑 BERT 前半的文字层；return_dict=True 可用 .last_hidden_state 取 [B,T,Dt]。
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        # 文本也取 CLS 得到 [B,E]；此时还处于 mode='text'，未与图像融合。
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
             
        # get momentum features
        with torch.no_grad():
            # 带 _m 的教师网络不经梯度更新，而是对学生参数做滑动平均。
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            # 教师也把图像 CLS 映射到 [B,E]，但此处不计算梯度。
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)
            # 当前 batch 的教师特征 + 历史队列：扩大候选集合。
            # 形状 [E,B+Q]，用于与学生的 [B,E] 相乘。
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)
            # 带 _m 的是动量教师文字编码器；它读相同文字，却用时间平均后的参数。
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            # 文本候选同样由当前 batch 与历史队列拼成 [E,B+Q]。
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            # [B,E] @ [E,B+Q] -> [B,B+Q]，每行是一张图对所有文本的分数。
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            # t2i = text to image；相同 [B,B+Q]，每行是一句文字的所有图候选。
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            # 当前批第 i 行的第 i 列是真配对；历史队列列没有本批的一热正标签。
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            # 当前 batch 的第 i 张图配第 i 段文字；队列列没有一热正例。
            sim_targets.fill_diagonal_(1)

            # alpha 混合教师的软概率与真实配对的一热标签，得到蒸馏目标。
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # 学生预测要接近上面教师与真实标签混合出的目标分布。
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        # 软标签交叉熵：逐候选求 target × log(预测概率)，再对 batch 平均。
        # log_softmax 沿每行候选归一化；与目标逐项相乘求和取负是软标签交叉熵。
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        # 两个方向平均，既训练“图找文”，也训练“文找图”。
        loss_ita = (loss_i2t+loss_t2i)/2

        # 当前教师特征进入队列，供后续 batch 充当额外对比候选。
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        # mode='fusion' 让文字 token 通过交叉注意力读取图像 patch。
        # encoder_embeds 是先前已经算出的文字 token 序列，避免重复文字嵌入。
        # encoder_hidden_states 是图像 patch 序列；fusion 层的文字 query 会读取它。
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            # bs=batch size，只表示当前本卡 batch；后面 ITM 从本批而非历史队列抽错配。
            bs = image.size(0)          
            # 只在当前 batch 抽难负例；队列中只有向量，缺少融合所需 token。
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            # 采负例时把正确配对的对角线概率清零，避免抽到真配对。
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        # *_neg 保存负例 token 序列，不是队列里的 256 维全局向量。
        image_embeds_neg = []
        for b in range(bs):
            # 根据相似度概率抽错图，越像正确图片越可能被抽到，因而更难。
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            # 对每张图也抽一段相似但不匹配的错误文字。
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        # 第 1 组文字是真文、第 2 组是错文；与下方图像顺序配合造两类错配。
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        # 图像顺序反过来：第 1 组错图配真文、第 2 组真图配错文，共 2B 对。
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        # 这 2B 个负配对仍需真正运行跨注意力，不能只凭全局相似度给 ITM 结论。
        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        # 正例和两组负例的融合 CLS 拼成 [3B,D]，统一送入二分类头。
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        # 真配对 B 个、两类难负配对共 2B 个，训练二分类匹配头。
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        
        ##================= MLM ========================##                
        # clone 复制整数 ID；否则 mask 的原地替换可能改坏原 text.input_ids。
        input_ids = text.input_ids.clone()
        # labels 留住遮挡前正确词，稍后未抽中的位置改成 -100 忽略。
        labels = input_ids.clone()

        # 与 labels 同形状，每格默认 0.15 遮挡概率；实际特殊 token 在 mask() 中排除。
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 随机遮住一部分文字；labels 保留原词，其余位置设为忽略值。
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            # 教师给被遮位置产生词表软概率，供学生蒸馏。
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        # 具体“硬答案 + 教师软词表”的混合损失在 xbert.BertForMaskedLM.forward 中实现。
        loss_mlm = mlm_output.loss

        # 返回顺序与实际计算顺序不同；训练脚本按此顺序解包并求和。
        return loss_mlm, loss_ita, loss_itm

        

    @torch.no_grad()    
    def copy_params(self):
        # 初始化时执行；@torch.no_grad() 让参数复制不进入反向传播计算图。
        for model_pair in self.model_pairs:           
            # zip 同步遍历学生与教师同序参数；param_m 的 m 是 momentum。
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        # 每次 forward 的教师参数 = momentum*旧教师 + (1-momentum)*当前学生。
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        # 多 GPU all_gather 后 batch_size 是所有进程合起来的样本数。
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        # ptr = pointer，表示环形队列下一次写入的起始列；int 取出 Python 整数。
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # 本批 [总B,E] 转置成 [E,总B]，覆盖从 ptr 起的相应列。
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        # input_ids 是可修改的词 ID 副本；targets 是原词监督副本；vocab_size 决定随机词范围。
        if masked_indices is None:                                       
            # bernoulli 按 probability_matrix 每格抽 0/1，再转布尔遮挡位置。
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        # PAD 补齐和 CLS 汇总位置不作为 MLM 预测题。
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # 剩余 20% 中取一半，即所有被选位置的约 10% 改成随机词，另约 10% 保持原词。
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
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
    # 每个进程预备一个同形状接收张量；get_world_size() 是参与训练的进程数。
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

