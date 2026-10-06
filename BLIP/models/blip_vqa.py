# 教学副本：下方保持原仓库可执行语句和已有注释，仅增加中文学习提示。
# 原文件：BLIP/models/blip_vqa.py。请配合 BLIP-从零重讲-04-任务与API.md 阅读。
# 本仓库的 MED 文件提供配置、文字/图文编码器、逐词答案解码器；三者不是同一个对象。
from models.med import BertConfig, BertModel, BertLMHeadModel
# 本仓库函数：建视觉编码器、建扩充特殊词的 tokenizer、按 checkpoint 载入权重。
from models.blip import create_vit, init_tokenizer, load_checkpoint

# PyTorch 负责张量创建/拼接/索引和设备迁移。
import torch
# nn 提供 Module 模型基类、Linear 等有参数网络层；此文件的 VQA 类继承 nn.Module。
from torch import nn
# F 是 functional；下面使用 F.softmax 把词分数变概率。
import torch.nn.functional as F
# BertTokenizer 类在本文件未直接调用，真正实例由 init_tokenizer() 创建。
from transformers import BertTokenizer
# np 是 NumPy 的惯用别名；底部 tile() 用 np.arange 和 np.concatenate 计算重排索引。
import numpy as np

class BLIP_VQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        # 先初始化父类 nn.Module，self.visual_encoder 等子层才能被 PyTorch 正确注册。
        super().__init__()
        
        # 左边元组拆包：视觉模型与其 token 宽度 Dv；VQA 默认图片边长较大。
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        # 取得 BERT 分词器并加入 [ENC]/[DEC]；下面问题和答案会用同一词表。
        self.tokenizer = init_tokenizer()  
        
        # med_config 是 JSON 文件路径；encoder_config 是解析后的可设置属性的配置对象。
        encoder_config = BertConfig.from_json_file(med_config)
        # 问句编码器的交叉注意力读图像 token，因此 K/V 输入宽度设置为 ViT 的 Dv。
        encoder_config.encoder_width = vision_width
        # 编码问题；传图像 token 后，各文字位置通过交叉注意力吸收图像信息。
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        # 再次读配置创建答案解码器的配置对象；与问题编码器是两套参数。
        decoder_config = BertConfig.from_json_file(med_config)        
        # LMHead 把每个解码位置映射到词表分数，训练时可计算下一词损失。
        self.text_decoder = BertLMHeadModel(config=decoder_config)          


    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):
        
        # image 是一批像素图 [B,3,H,W]；image_embeds 是全部视觉 token [B,P,Dv]。
        image_embeds = self.visual_encoder(image) 
        # 图像没有 PAD；size()[:-1] 得 [B,P]，ones 全 1 表示 token 均有效。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        # 这里 question 从原始字符串列表变成 tokenizer 输出对象；后面用 question.input_ids。
        # question 参数原是 B 个字符串；此行之后同名 question 指 tokenizer 输出对象，含 IDs/mask。
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(image.device) 
        # 问题走“编码”路径，所以首 token 用 encoder 专用标记。
        # 把每句首位换成 [ENC]，表明这是供图文融合的问句编码，而非答案生成。
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        # train 是本函数传入的布尔开关；不要与 model.train() 混淆，后者切换 Dropout 等模式。
        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            # answer 参数原是全部候选正确答案字符串；此行后变成含 input_ids/mask 的 tokenizer 对象。
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            # 答案解码用 [DEC] 起始 token；bos=beginning of sequence，序列开头。
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            # 答案解码目标忽略 padding，-100 不参与交叉熵。
            # 把 PAD 标签改为 -100，CrossEntropyLoss 忽略这些位置；输入的 answer.input_ids 仍保留 PAD ID。
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            # 把问句 token 送入文字编码器；下面 encoder_hidden_states=image_embeds 使它同时看图片。
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            # 这个列表稍后存每个答案配套的“看过图的问句向量”，不是原始问题字符串。
            # 稍后列表中每项都是一个问题的融合后 token 向量 [T,Dt]，按其答案数重复。
            question_states = []                
            # 每份重复的问题向量都要配一份对应文字 0/1 mask；顺序必须与答案列表一致。
            question_atts = []  
            # 一个问题可能有多个可接受答案；每个答案都要配同一份问题特征。
            # 参数 n 是每个问题对应的答案数量列表；循环变量 n 是当前第 b 个问题的答案数。
            # 外层参数 n 是各问题答案数列表；循环变量 n 是第 b 个问题的答案数，命名重复要留意。
            for b, n in enumerate(n):
                # 第 b 个问题的融合状态复制 n 次；一题多答案时每个答案配同一道题。
                question_states += [question_output.last_hidden_state[b]]*n
                # 第 b 个问题的有效词 mask 同步复制 n 次，与上面的 question_states 一一对应。
                question_atts += [question.attention_mask[b]]*n                
            # 把按答案复制的问句特征列表叠为 [总答案数,T,Dt]，与答案批量一一对齐。
            # stack 新增最前面的答案 batch 维，形成 [总答案数,T,Dt]。
            question_states = torch.stack(question_states,0)    
            # 把各答案对应的问题 mask 堆为 [总答案数,T]。
            question_atts = torch.stack(question_atts,0)     

            # 答案解码器输入标准答案 token，并通过 encoder_hidden_states 读取上面“看过图的问题”。
            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_states,
                                              encoder_attention_mask = question_atts,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )      
            
            # VQA 的多个答案可有不同人工标注分数，故逐答案加权后再求和。
            # reduction=none 返回每条答案的 loss；weights 给人工认可度高的答案更大权重。
            loss = weights * answer_output.loss
            # 先把所有候选答案的加权损失求和，再除以本批问题数 B。
            # 先跨全部答案求和，再除问题数 B，而不是除答案总数。
            loss = loss.sum()/image.size(0)

            # 训练分支返回一个标量，外层训练脚本调用 backward() 和 optimizer.step()。
            return loss
            

        else: 
            # 把问句 token 送入文字编码器；下面 encoder_hidden_states=image_embeds 使它同时看图片。
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            # generate 直接逐词生成答案；rank 则在给定候选答案中排序。
            # 生成模式直接逐词给出新答案；下面 rank 分支则必须有预先给出的候选答案。
            # 推理生成模式从 [DEC] 逐词写答案；另一个 rank 模式是从已有候选答案里选。
            if inference=='generate':
                # beam 是束搜索中并行保留的候选句，3 表示每题最多同时搜索 3 个候选路径。
                num_beams = 3
                # 每个看过图的问题状态复制 3 次，供 3 条束各自读取同一问题条件。
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
                # 按复制后的 [B×beam,T] 形状建 mask；这里源码令所有问题位置有效。
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                # 字典随后用 **model_kwargs 展开，传给 generate() 作为每步都可读取的问句条件。
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                # 每道题生成只从一个 [DEC] ID 开始；full 按给定形状填相同整数。
                bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
                
                # Hugging Face 生成循环：反复调用解码器，按 beam search 接下一个词直到结束/长度限制。
                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
                # 收集每道题最终可读的答案字符串；outputs 目前还是整数 token ID 序列。
                answers = []    
                # outputs 中每条是一道题生成的 token ID 序列；逐条转回自然语言。
                for output in outputs:
                    # 把一条生成的 ID 序列转回文字，并跳过 [DEC]/[SEP]/[PAD]。
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                # 生成模式输出 Python 答案字符串列表，顺序与输入问题一致。
                return answers
            
            # rank=排名；只在传入候选答案对象 answer 时才可走这条分支。
            elif inference=='rank':
                # rank_answer 返回每道问题选中的候选答案序号，不返回已解码字符串。
                # 把看过图的问题向量、问题 mask 与候选答案 IDs/mask 送入两阶段排序。
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) 
                # rank 模式返回候选答案列表中的整数下标；调用者再查原答案文本。
                return max_ids
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        # num_ques=number of questions；此函数 question_states 是已看图的问题向量，首维是问题数。
        num_ques = question_states.size(0)
        # 从第一个候选答案取 [DEC] ID，复制成每道问题一个起始 token，形状 [题数,1]。
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        # 只喂答案起始 token，先预测答案的第一个实际词，用于便宜地筛候选。
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        # 取起始位置对整个词表的原始分数，形状 [题数,词表大小]。
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        # 每个候选答案第 1 个位置是首个实际词；第 0 个位置是 [DEC]。
        answer_first_token = answer_ids[:,1]
        # 先把词表分数转概率，再只抽出候选答案各自首词概率；形状 [题数,候选数]。
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        # 先按答案第一个词的概率筛选 k 个；topk_ids 是候选列表中的位置。
        # topk 返回最高 k 个概率及其候选答案下标；只是粗筛，不是最终整句概率。
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        # 收集每道题 top-k 候选答案的 PAD mask；顺序要与 input_ids 一致。
        input_atts = []
        # b 是问题序号，topk_id 是这道题选中的 k 个候选答案下标。
        for b, topk_id in enumerate(topk_ids):
            # 按当前问题的 topk_id，从候选答案矩阵挑出 k 条完整 token 序列。
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            # 用相同下标挑 mask，确保每条答案的 PAD 位置匹配其 IDs。
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        # 把所有问题的 k 条答案沿 batch 维拼成 [题数×k,答案长度]。
        input_ids = torch.cat(input_ids,dim=0)  
        # 把答案 mask 按完全相同的问题/候选顺序拼接。
        input_atts = torch.cat(input_atts,dim=0)  

        # 完整答案重打分时 PAD 标签置 -100，不参与逐词交叉熵。
        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        # tile 复制并重排问题向量，使顺序与 [每题 k 条答案] 一一对应。
        question_states = tile(question_states, 0, k)
        # 问题 mask 同样复制 k 份并按题分组；这里同名变量已变成扩充后的 batch。
        question_atts = tile(question_atts, 0, k)
        
        # 计算每个候选答案的完整逐词损失；reduction=none 保留每条答案单独的值。
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        # 完整候选答案的语言模型损失越小，负损失作为得分越大。
        # 交叉熵是负对数似然，取负变为对数概率和；数值越大表示答案越合适。
        log_probs_sum = -output.loss
        # 把扁平的 [题数×k] 得分重排为每题一行、每行 k 个候选。
        log_probs_sum = log_probs_sum.view(num_ques,k)

        # 每道问题在 k 个候选中找最大完整答案得分的位置。
        # argmax 找每道题 k 个候选里完整答案得分最大的局部位置。
        max_topk_ids = log_probs_sum.argmax(dim=1) 
        # 将 top-k 内的局部位置映回原候选答案列表下标；前一索引用布尔值选全部有效题。
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        # rank 模式返回候选答案列表中的整数下标；调用者再查原答案文本。
        return max_ids
    
    
def blip_vqa(pretrained='',**kwargs):
    # kwargs 是额外命名参数字典；** 展开后传给 VQA 模型构造器。
    model = BLIP_VQA(**kwargs)
    # 仅当给出已有权重路径/URL 时才调用 load_checkpoint；否则随机初始化结构。
    if pretrained:
        # pretrained 非空时装载已有权重，msg 保存缺失/多余层的诊断信息。
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  


def tile(x, dim, n_tile):
    # tile 的 x 可为问题状态或问题 mask；init_dim 是要复制的那一维原长度。
    init_dim = x.size(dim)
    # 为 x 的每一维建立重复次数，初始全部 1；x.dim() 返回张量维数。
    repeat_idx = [1] * x.dim()
    # 只把目标维设为 n_tile 次，其余维度不重复。
    repeat_idx[dim] = n_tile
    # * 将列表拆为多个位置参数；repeat 先按块复制张量。
    x = x.repeat(*(repeat_idx))
    # NumPy 构造交错索引：让同一道题的 k 份复制品相邻，而不是所有题按整批重复。
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    # index_select 按交错索引重排目标维；索引也必须搬到 x 所在设备。
    return torch.index_select(x, dim, order_index.to(x.device))
