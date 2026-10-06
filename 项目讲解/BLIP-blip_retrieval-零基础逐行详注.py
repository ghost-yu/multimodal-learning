# 教学副本：下方保持原仓库可执行语句和已有注释，仅增加中文学习提示。
# 原文件：BLIP/models/blip_retrieval.py。请配合 BLIP-从零重讲-04-任务与API.md 阅读。
# 本仓库 med.py 的 BertConfig 保存网络超参数；BertModel 将文字 token 编码，并可通过交叉注意力读图。
from models.med import BertConfig, BertModel
# Hugging Face 分词器类；本文件未直接调用，实际经 init_tokenizer() 创建带 [ENC]/[DEC] 的实例。
from transformers import BertTokenizer

# PyTorch 主模块：创建张量、分布式收集与自动求导都使用 torch。
import torch
# nn 是 neural network 命名空间：Module 基类、Linear 线性层、Parameter 可训练标量。
from torch import nn
# F 是 functional 的惯用别名，调用 normalize、softmax、log_softmax 和 cross_entropy。
import torch.nn.functional as F

# 三个本仓库函数分别创建 ViT、创建扩充的 BERT 分词器、从路径或 URL 装载权重。
from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        # 先执行 nn.Module 的构造器，之后 self.xxx 的子层和 buffer 才会被 PyTorch 管理。
        super().__init__()
        
        # 一次函数返回两个值：ViT 模型对象与图像 token 宽度 Dv；左边用元组拆包接收。
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        # tokenizer 把字符串转 ID 和 0/1 mask；之后多次由 self.tokenizer(caption,...) 调用。
        self.tokenizer = init_tokenizer()   
        # 同名局部变量原是 JSON 路径字符串，此行之后变为配置对象；不要与路径混淆。
        med_config = BertConfig.from_json_file(med_config)
        # 将上面 create_vit 返回的 Dv 告诉文字交叉注意力，便于把图像 K/V 投影到文字宽度。
        med_config.encoder_width = vision_width
        # 创建文字编码器；text 模式只读文字，图文模式还可读 ViT 的全部图像 token。
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        # 从实际 BERT 配置读出每个文字 token 的维度 Dt，典型 Base 为 768。
        text_width = self.text_encoder.config.hidden_size
        
        # proj=projection；整图 CLS 由 Dv 维投影到共同 E 维，对比学习才可比较。
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        # 整句 CLS 由 Dt 维投影到同样的 E 维；Linear 只变输入最后一维。
        self.text_proj = nn.Linear(text_width, embed_dim)

        # ITM=Image-Text Matching；融合 CLS 输出 2 个 logits，分别代表不匹配/匹配。
        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        # _m=momentum（动量教师）；再次创建同规格 ViT，随后 copy_params 复制学生初值。
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        # 教师的图像投影层与学生形状相同，但参数会走滑动平均更新。
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # 教师文字编码器使用同一结构配置，通常仅用于提供稳定的对比目标。
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        # 教师文字 CLS 投影到 E 维，使教师图文特征能够相互点积。
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        # model_pairs 中每一项按 [学生模块,教师模块] 排列，供下面 zip 参数和 EMA 更新。
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        # 先把学生参数逐项复制给教师，并设置教师不直接计算参数梯度。
        self.copy_params()

        # create the queue
        # 每列是历史图片 E 维向量；buffer 随模型保存和搬设备，但不由优化器直接学习。
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        # 历史文字队列与图片队列列序对齐；randn 先给随机值，后来由真实教师特征覆盖。
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        # 每个队列列对应图片 ID；-100 是初始哨兵 ID，避免空队列误认成某张真实图片。
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        # ptr=pointer，记录下次覆盖哪一列；long 类型适合做整数索引。
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        # dim=0 沿每一列的 E 个分量做 L2 归一化，因为列才是单个样本向量。
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # 文字队列也按列归一化；之后向量点积对应余弦相似度。
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        # 把构造参数 Q 保存到模型实例，供循环队列入队时检查长度和回绕。
        self.queue_size = queue_size
        # 保存 EMA 系数，默认 0.995；教师保留 99.5% 旧值、吸收 0.5% 学生新值。
        self.momentum = momentum
        # temp=temperature，空形状 [] 是标量张量；Parameter 表示训练中可学习。
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        # rank 在分布式术语中是 GPU 进程编号；True 时从所有进程抽负例。
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption, alpha, idx):
        # 该代码块暂不建立反向传播图；裁温度、教师目标或抽样权重无需对其自身求导。
        with torch.no_grad():
            # clamp_ 将标量限制在区间内；末尾下划线表示原地修改，防止极端温度。
            self.temp.clamp_(0.001,0.5)
        
        # image 是当前 batch 的像素张量 [B,3,H,W]；embeds 是完整图像 token [B,P,Dv]。
        image_embeds = self.visual_encoder(image) 
        # size()[:-1] 去掉特征宽度得到 [B,P]；每个图像 token 有效，故 mask 全 1。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        # 第 0 个 CLS 汇总整图；经 Linear 到 E 维，再按最后一维 L2 归一化，形状 [B,E]。
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        
        # caption 是字符串列表；从此行起 text 是分词结果对象，含整数 input_ids 与 attention_mask，各 [B,35]。
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        # 输入 tokenizer 的两个张量；下面 mode=text 跳过图像交叉注意力，输出纯文字表示。
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        # text_output 是上一行纯文本编码结果；取文字 CLS 后得归一化整句向量 [B,E]。
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)        
        
        ###============== Image-text Contrastive Learning ===================###
        # idx 是每条图文所属图片的 ID；view(-1,1) 从 [B] 改 [B,1] 以便广播比较。
        # idx 来自数据集的图片 ID 而非 batch 位置；由 [B] 改 [B,1] 以便广播比较。
        idx = idx.view(-1,1)
        # 前 B 列是当前批的图片 ID，后 Q 列是历史队列 ID；一共 [1,B+Q]。
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        # 广播比较同 ID，即使不同 caption 指向同一图，也都应标为正例。
        # 同图片 ID 的多段描述都为正；eq 产生布尔矩阵，再转成可除法的浮点 0/1。
        pos_idx = torch.eq(idx, idx_all).float()       
        # 一行可能有多个真配对，除以该行正例数量后目标概率和为 1。
        # 一行可有多个正例；除正例个数，使每行硬目标概率合为 1。
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        
        # get momentum features
        # 该代码块暂不建立反向传播图；裁温度、教师目标或抽样权重无需对其自身求导。
        with torch.no_grad():
            # 此处教师四组参数按 EMA 公式更新，而不是通过当前损失反向传播。
            self._momentum_update()
            # _m 表示动量教师；输入仍是同一批 image，输出教师完整图像 token。
            image_embeds_m = self.visual_encoder_m(image) 
            # 教师图像 CLS 投影到 E 维，用于软目标和入队；与学生 image_feat 区分。
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            # 当前教师 [B,E] 转置成 [E,B]，与队列 [E,Q] 按候选列拼成 [E,B+Q]。
            image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            # 教师只看同一批文字 ID；text 仍指本次 tokenizer 对象，不是原始 caption 字符串。
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            # 取教师文字 CLS，投影并归一化成 [B,E]，与 image_feat_m 对应。
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            # 教师本批文字与过去 Q 条文字组成候选列；_all 指当前加历史而非全数据集。
            text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            # i2t=image-to-text，@ 做 [B,E]@[E,B+Q] 得 [B,B+Q]；_m 表示教师分数。
            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            # t2i=text-to-image，行是文字查询，列是图像候选；除温度控制 softmax 尖锐度。
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp   

            # alpha 混合教师软分布与同 ID 硬标签；softmax 沿候选列归一化。
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # 反向检索也用同 ID 正例矩阵，但教师分数改为文字查图像。
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # 学生图像向量对教师当前加队列文字候选打分；梯度可回到学生投影与 ViT。
        sim_i2t = image_feat @ text_feat_m_all / self.temp 
        # 学生文字向量对教师图像候选打分；仍是 [B,B+Q]。
        sim_t2i = text_feat @ image_feat_m_all / self.temp 
                             
        # log_softmax 得候选对数概率；乘目标后逐行求和取负，再跨 B 条样本平均。
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        # 与上一行同公式，但查询由文字发出；两项都为单个标量。
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        # ITA=image-text alignment；把两个检索方向的对比损失平均。
        loss_ita = (loss_i2t+loss_t2i)/2
        
        # idxs 是全部 GPU 的当前批图片 ID；必须与全局 gathered 特征采用相同顺序。
        idxs = concat_all_gather(idx)
        # 入队的是动量教师的两侧向量及相同顺序的图片 ID，供未来 batch 使用。
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)        

        ###============== Image-text Matching ===================###
        # text 仍是前面 tokenizer 结果；复制后改首 token，不影响原始 input_ids。
        encoder_input_ids = text.input_ids.clone()
        # 首 token 改为 [ENC] 的整数 ID，提示文字编码器进入图文融合路径。
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        # bs 是本 GPU batch 大小 B，不是 all_gather 后的全局批量。
        bs = image.size(0)
        # 真配对的文字读取对应图片 token；输出每个文字位置的融合状态。
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )  
        
        
        # True 时负样本可来自所有 GPU；False 时只在本 GPU batch 中抽。
        if self.negative_all_rank:    
            # compute sample similarity
            # 该代码块暂不建立反向传播图；裁温度、教师目标或抽样权重无需对其自身求导。
            with torch.no_grad():                
                # 跨 GPU 比较每条查询的图片 ID 与全部进程 ID；同 ID 候选均不能作为负例。
                mask = torch.eq(idx, idxs.t())

                # world 指全部进程；收集整体图像向量只用于选难负例，无需跨卡梯度。
                image_feat_world = concat_all_gather(image_feat)
                # 收集全部进程的整句向量；列顺序与 idxs 对齐。
                text_feat_world = concat_all_gather(text_feat)

                # 跨 GPU 难负例：本地 B 张图片对全部进程文字做点积；矩阵列数为全局 batch。
                sim_i2t = image_feat @ text_feat_world.t() / self.temp 
                # 反方向本地文字对全局图像打分；world 特征只用于计算抽样概率。
                sim_t2i = text_feat @ image_feat_world.t() / self.temp 

                # 把每行图找文相似度转成非负概率，越相似的错误句子越可能被抽到。
                weights_i2t = F.softmax(sim_i2t,dim=1)
                # 把同 ID 正例抽样权重原地设 0，否则可能误抽真正配对当负例。
                weights_i2t.masked_fill_(mask, 0)            

                # 把文找图相似度转抽样概率；下一行 mask 排除所有同 ID 图片。
                weights_t2i = F.softmax(sim_t2i,dim=1)
                # 文字查图方向也排除所有同 ID 图片；不能只排对角线。
                weights_t2i.masked_fill_(mask, 0)     

            # world 指所有进程；这里收集完整图片 token 并保留可反传的梯度路径。
            # 这里需要完整图像 token 做 ITM，且抽中的跨卡图像要有梯度，所以使用带梯度 gather。
            image_embeds_world = all_gather_with_grad(image_embeds) 

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []    
            # b 是本 GPU 内第 b 个查询；本循环为每个查询挑一个相似但不匹配的负例。
            for b in range(bs):
                # 按第 b 句文字的图像候选权重抽一个位置；item 把单元素张量变 Python 整数。
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            # 列表中每项是 [P,Dv] 完整错图 token；stack 新增 batch 维成 [B,P,Dv]。
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from all ranks) for each image
            # 跨卡收集图文融合用文字 ID；整数 ID 本身不需梯度。
            input_ids_world = concat_all_gather(encoder_input_ids)
            # 同时收集各段文字的有效位 mask，必须与文字 ID 顺序相同。
            att_mask_world = concat_all_gather(text.attention_mask)        

            # 列表收集每张真图配到的错误文字 ID；稍后 torch.stack 形成 [B,T]。
            text_ids_neg = []
            # 同步收集负文字的注意力 mask，保证与 text_ids_neg 按下标一一对应。
            text_atts_neg = []
            # b 是本 GPU 内第 b 个查询；本循环为每个查询挑一个相似但不匹配的负例。
            for b in range(bs):
                # 按第 b 张真图的错误文字权重抽位置；同 ID 正例此前权重已置 0。
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                
        else:
            # 该代码块暂不建立反向传播图；裁温度、教师目标或抽样权重无需对其自身求导。
            with torch.no_grad():                
                # 本 GPU 内比较两条样本图片 ID；重复 ID 都为正例，抽样时要排除。
                mask = torch.eq(idx, idx.t())
                
                # 本 GPU 路径：只对本地 B 个文字候选打分，不含其他 GPU。
                sim_i2t = image_feat @ text_feat.t() / self.temp 
                # 本 GPU 路径：只对本地 B 个图像候选打分；结果 [B,B]。
                sim_t2i = text_feat @ image_feat.t() / self.temp 

                # 把每行图找文相似度转成非负概率，越相似的错误句子越可能被抽到。
                weights_i2t = F.softmax(sim_i2t,dim=1)
                # 把同 ID 正例抽样权重原地设 0，否则可能误抽真正配对当负例。
                weights_i2t.masked_fill_(mask, 0)            

                # 把文找图相似度转抽样概率；下一行 mask 排除所有同 ID 图片。
                weights_t2i = F.softmax(sim_t2i,dim=1)
                # 文字查图方向也排除所有同 ID 图片；不能只排对角线。
                weights_t2i.masked_fill_(mask, 0)     

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            # b 是本 GPU 内第 b 个查询；本循环为每个查询挑一个相似但不匹配的负例。
            for b in range(bs):
                # 按第 b 句文字的图像候选权重抽一个位置；item 把单元素张量变 Python 整数。
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            # 列表中每项是 [P,Dv] 完整错图 token；stack 新增 batch 维成 [B,P,Dv]。
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            # 列表收集每张真图配到的错误文字 ID；稍后 torch.stack 形成 [B,T]。
            text_ids_neg = []
            # 同步收集负文字的注意力 mask，保证与 text_ids_neg 按下标一一对应。
            text_atts_neg = []
            # b 是本 GPU 内第 b 个查询；本循环为每个查询挑一个相似但不匹配的负例。
            for b in range(bs):
                # 按第 b 张真图的错误文字权重抽位置；同 ID 正例此前权重已置 0。
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])            
            
        # 把 B 段抽到的错文字 ID 堆成 [B,T]；负文字与正图片按行对应。
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        # 将与负文字一一对应的 0/1 mask 堆成 [B,T]，避免 PAD 被注意。
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        # 前 B 行真文字配错图；后 B 行错文字配真图；合计 2B 行。
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        # mask 的行顺序与 text_ids_all 完全一致，避免错误遮住词。
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        # 前 B 行错图，后 B 行真图，与上面 [真文字,错文字] 两段交叉配成负例。
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        # 正负图片 token 均无 padding，所以把同一批全 1 mask 拼两次。
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        # 一次跑完 2B 个错误图文配对；融合后的第 0 个文字 token 用于 ITM 分类。
        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                         
          

        # vl=vision-language；B 个正例加 2B 个负例的融合 CLS，形状 [3B,Dt]。
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        # Linear(Dt,2) 给每对图文两个未归一化类别分数，形状 [3B,2]。
        vl_output = self.itm_head(vl_embeddings)            

        # 前 B 个标签 1 是匹配，后 2B 个标签 0 是不匹配；long 是交叉熵要求的整数类别型。
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        # cross_entropy 接原始 logits 和整数标签，内部做 log_softmax，返回平均分类损失。
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        # 以二元组返回对比损失和图文匹配损失；检索微调没有 LM 生成损失。
        return loss_ita, loss_itm 
 

    @torch.no_grad()    
    def copy_params(self):
        # 每个 model_pair 都是 [学生模块,教师模块]；下面逐参数配对。
        for model_pair in self.model_pairs:           
            # zip 按顺序并行迭代两组参数；param 是学生，param_m 是动量教师。
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                # 教师不由 loss.backward 更新；后面的 _momentum_update 直接写其参数值。
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        # 每个 model_pair 都是 [学生模块,教师模块]；下面逐参数配对。
        for model_pair in self.model_pairs:           
            # zip 按顺序并行迭代两组参数；param 是学生，param_m 是动量教师。
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # EMA 公式：教师新=旧教师×m+当前学生×(1-m)，不是计算优化器梯度。
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        # 复数 feats 表示所有 GPU 共同的批；单数 feat 是调用者本 GPU 批。
        image_feats = concat_all_gather(image_feat)
        # 与图片教师特征按相同全局批顺序收集，保证同列图文配对。
        text_feats = concat_all_gather(text_feat)
        

        # 这是 all_gather 后的全局批量，不是本 GPU 的 bs；队列一次覆盖这么多列。
        batch_size = image_feats.shape[0]

        # 从单元素指针张量取 Python 整数，供下方切片起点使用。
        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # 队列每列一条样本，而新特征是 [全局B,E]，转置后写入 [E,全局B]。
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        # 相同列范围写教师文字向量，图片/文字/ID 三条队列始终对齐。
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        # 队列特征和 ID 在同一列位置一起覆盖，之后才能按 ID 判定多个正例。
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        # 取余让写入位置到队列末尾后回到 0，形成循环队列。
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        # 把更新后的 Python 整数写回 buffer，下次 batch 从新位置开始。
        self.ptr_queue[0] = ptr  


def blip_retrieval(pretrained='',**kwargs):
    # kwargs 是额外命名参数字典，** 展开后传给构造器；model 是创建好的检索模型。
    model = BLIP_Retrieval(**kwargs)
    # 传入非空预训练路径/URL 才载入权重；空字符串只建模型结构。
    if pretrained:
        # pretrained 是路径/URL；函数返回载入权重后的模型和缺失/多余键报告。
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        # 列出未从 checkpoint 载入的层名；新任务头缺失可能正常，主干大面积缺失需核对。
        print(msg.missing_keys)
    # 工厂函数返回已构造、可选已载入预训练权重的检索模型。
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # 每个进程预分配 world_size 个与本地 tensor 同形状的接收张量。
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    # 所有进程相互交换 tensor；此普通 all_gather 不为跨进程数据保留反向传播梯度。
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # 沿 batch 维拼各进程张量，得到全局批；进程顺序与 ID 收集顺序要相同。
    output = torch.cat(tensors_gather, dim=0)
    # 返回沿 batch 维合并后的全部进程张量；普通 gather 不保留跨进程梯度。
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        # 带梯度 GatherLayer 的前向也先为每个 GPU 结果分配同形接收容器。
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        # 前向收集各进程 x；自定义 backward 将在之后明确给梯度回传规则。
        torch.distributed.all_gather(output, x)
        # 自定义 autograd 前向返回每进程一个张量组成的元组，供后续沿 batch 拼接。
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        # grads 是各输出位置的梯度元组；stack 新增 GPU/进程维。
        all_gradients = torch.stack(grads)
        # 把各进程对这些输出的梯度求和，使被其他进程使用的样本也收到梯度。
        torch.distributed.all_reduce(all_gradients)
        # 每个进程只取自己输入 x 对应位置的梯度返回给自动求导。
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    # world_size 是进程/GPU 总数；需先初始化 torch.distributed 进程组。
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        # world_size=1 时无需跨进程通信，原样返回输入张量，并保留正常梯度。
        return tensors

    # 自定义 autograd.Function 通过 apply 进入前向，同时登记对应 backward。
    tensor_all = GatherLayer.apply(tensors)

    # 把所有进程图片 token 沿 batch 维拼在一起；梯度路径仍连着各进程输入。
    return torch.cat(tensor_all, dim=0)
