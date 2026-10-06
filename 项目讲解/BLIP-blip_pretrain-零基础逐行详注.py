# 教学副本：保持官方仓库现有可执行语句不变，只增补给初学者的中文解释。
# 结构对照：原始文件 BLIP/models/blip_pretrain.py；建议同时阅读 项目讲解/BLIP-从零重讲 系列。
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
# med.py 提供改造后的 BERT：既能单独读文字，也能通过交叉注意力读取图像。
from models.med import BertConfig, BertModel, BertLMHeadModel
# tokenizer 把字符串转成模型能处理的整数 token ID。
from transformers import BertTokenizer
# 此处还导入整个 transformers 包，只为了下一行调用其 logging 接口。
import transformers
# 仅显示 Transformers 的错误，避免加载预训练权重时输出大量提示。
transformers.logging.set_verbosity_error()

# torch 负责张量运算；nn 负责可训练的网络层；F 提供归一化、softmax、交叉熵。
import torch
# nn 是 neural network：Module、Linear、Parameter 等可训练结构都从这里取。
from torch import nn
# F 是 functional 的别名；normalize、softmax、log_softmax、cross_entropy 是函数调用。
import torch.nn.functional as F

# 复用同仓库里创建视觉编码器、tokenizer 的函数；load_checkpoint 当前文件未使用。
from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_Pretrain(nn.Module):
    """BLIP（Bootstrapping Language-Image Pre-training）预训练模型。

    论文称 ITC = Image-Text Contrastive（图文对比）；此代码把同一目标命名为
    ITA = Image-Text Alignment（图文对齐）。ITM = Image-Text Matching（图文匹配）；
    LM = Language Modeling（语言建模、预测下一个词）。
    术语和库函数的输入/输出见项目讲解/04-BLIP-术语与库函数详解.md。
    """
    def __init__(self,                 
                 med_config = 'configs/bert_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        # 先初始化 nn.Module，否则子模块不能正确注册成可训练参数。
        super().__init__()
        
        # patch 是图片的小方块；token 是模型处理的一个位置。ViT 输出 [B,P,Dv]，
        # B=批量大小，P=图像 token 数（包含 CLS），Dv=每个图像 token 的维数。
        # 左侧同时接收两个返回值：visual_encoder 是模型对象；vision_width 是每个图像 token 的维数。
        # 这里的 self.visual_encoder 在整个 forward 中反复使用，输入批量图片，输出全部图像 token。
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        
        # vit 是构造参数，字符串 base/large 决定加载哪一种视觉初始权重。
        if vit=='base':
            # Base 视觉编码器从 DeiT 权重初始化；第一次运行需要下载权重。
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            # checkpoint 是外层字典，真正的层参数放在 model 键下。
            state_dict = checkpoint["model"]     
            # strict=False 允许 BLIP 额外的层没有出现在 DeiT 检查点里。
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
        elif vit=='large':
            # Large 使用 timm 提供的 ImageNet-21k 预训练 ViT 权重。
            from timm.models.helpers import load_custom_pretrained
            # default_cfgs 是 timm 中不同视觉模型预训练设置的字典。
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder,default_cfgs['vit_large_patch16_224_in21k'])        
               
        # 在 BERT tokenizer 基础上增加 [ENC] 和 [DEC] 特殊 token。
        self.tokenizer = init_tokenizer()   
        # JSON 文件说明 BERT 隐藏层等结构参数。
        # BertConfig 是配置对象而非神经网络；from_json_file 把 JSON 里的层数、头数、隐藏维等读入对象。
        encoder_config = BertConfig.from_json_file(med_config)
        # 交叉注意力读图时，图像 token 的维度必须告诉文字编码器。
        # encoder_width 告诉文字交叉注意力：图像 K/V 的输入宽度是多少；来源是上一行 create_vit 的第二个返回值。
        encoder_config.encoder_width = vision_width
        # 文字编码器先加载 bert-base-uncased 的语言参数。
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        # tokenizer 多了特殊 token，词向量表也必须扩到同样大小。
        # len(tokenizer) 是扩充特殊 token 之后的词表大小；此调用把可训练词向量表扩到这个大小。
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        # BERT 输出的每个文字 token 的维数；Base 通常是 768。
        text_width = self.text_encoder.config.hidden_size
        
        # CLS = Classification token（序列第 0 个汇总位置）；两边整体表示
        # 通过 nn.Linear（可训练线性层）投影到相同 embed_dim，才能做点积比较。
        # vision_proj 的 proj=projection（投影）；将图像 CLS 从 Dv 维映射到共同空间 E 维。
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        # text_proj 与 vision_proj 的输出宽度相同，所以图文整体向量才能做矩阵乘法比较。
        self.text_proj = nn.Linear(text_width, embed_dim)

        # 融合后的 CLS 经线性层得到“不匹配/匹配”两个类别分数。
        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        # 后缀 _m 表示 momentum（动量教师），不是另一个训练任务。
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)
        # _m=momentum（动量教师）；与学生层同形，下面 copy_params 先复制初值。
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # 教师文字编码器用与学生相同的 encoder_config，但拥有独立参数对象。
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)      
        # _m 文字投影层同样由 Dt 到 E；初始值由 copy_params 从学生 text_proj 复制。
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        # 每一对依次是学生网络和动量教师网络；EMA（指数滑动平均）更新教师，
        # 公式是“教师新值 = momentum×教师旧值 + (1-momentum)×学生新值”。
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        # 教师初始参数复制学生参数，此后用指数滑动平均更新。
        self.copy_params()

        # create the queue
        # register_buffer：张量随 state_dict 保存、随 model.to(device) 搬设备，
        # 但优化器不会把它当作参数。队列只存过去 batch 的低维整体特征。
        # image_queue 的每一列是一个 E 维历史图像特征；register_buffer 随模型保存和搬设备，但优化器不直接更新。
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        # 指针说明下一批特征要覆盖队列的哪一段。
        # ptr=pointer（指针），只存下次写入队列的列号；long 是 PyTorch 的 64 位整数类型。
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        # 每列代表一个样本，长度归一后点积才能当作余弦相似度。
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # 文字队列每一列也是一个样本，所以同样沿第 0 维归一化。
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        # 保存队列容量 Q 和指数滑动平均系数，后面的成员方法通过 self 读取它们。
        self.queue_size = queue_size
        # momentum 在后面 _momentum_update 再出现，仍是构造器传入的 EMA 系数。
        self.momentum = momentum
        # 可学习温度系数控制相似度 softmax 的尖锐程度。
        # temp=temperature（温度）；空形状 [] 表示一个标量张量，nn.Parameter 使它成为可学习参数。
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        # create the decoder
        # 解码器与文字编码器基于同一 BERT 结构，但使用生成式注意力。
        decoder_config = BertConfig.from_json_file(med_config)
        # 解码器做文字看图的交叉注意力，因此其图像输入宽度也是 vision_width。
        decoder_config.encoder_width = vision_width        
        # 解码器用于“看图像并预测下一个词”，区别于上面的文本编码器。
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        # 解码器的词表也必须能表示新增的 [DEC]/[ENC] token。
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        # 在可对应的位置共享编码器/解码器权重，减少参数且同步初始化。
        tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')
        
        
    def forward(self, image, caption, alpha):
        """输入图片 [B,3,H,W]、B 条字符串和教师混合权重，返回 3 个标量 loss。"""
        # caption 此时还是字符串列表；下面先 tokenizer，再计算三个损失。
        # no_grad 使 clamp_ 不进入反向传播图；_ 后缀表示原地修改。
        with torch.no_grad():
            # 避免温度太小导致相似度过大，或太大导致区别不明显。
            self.temp.clamp_(0.001,0.5)
        
        # ViT 把整批图片编码为 [B,P,Dv]：P 是图像 token 数。
        # image 是 forward 输入的一批像素图；image_embeds 是全部视觉 token [B,P,Dv]，不是图片文件。
        image_embeds = self.visual_encoder(image) 
        # 图像 token 没有 padding，因此对应注意力掩码全为 1，形状 [B,P]。
        # image_embeds.size()[:-1] 去掉最后的特征维，留下 [B,P]；全 1 表示每个图像 token 有效。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        # 图像 CLS -> 共同向量空间的整图表示 [B,E]。
        # [:,0,:] 取每张图第 0 个 CLS；Linear 变成 E 维；normalize 使每条向量长度约为 1。
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        
        # BertTokenizer.__call__ 把 B 条字符串变成整数 input_ids [B,30] 和
        # attention_mask [B,30]；padding 补短句，truncation 截长句；1 是有效词，0 是 PAD。
        # caption 是原始字符串列表；本语句之后 text 是 tokenizer 输出对象，含 input_ids 和 attention_mask。
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                              return_tensors="pt").to(image.device)  
        # mode='text' 只读文字，不传入图像 token。
        # text.input_ids 是整数词序列；attention_mask 中 1 是有效 token，0 是 PAD；mode=text 仅编码文字。
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        # 文本 CLS -> [B,E]；这一步是纯文本编码，尚未图文融合。
        # text_output 来源于上一段纯文本编码；取文字 CLS，经投影和归一化得到整句 E 维向量。
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)
             
        # get momentum features
        # 教师只生成稳定目标，不用反向传播训练；学生稍后拟合这些目标。
        with torch.no_grad():
            # 先按“旧教师 × momentum + 新学生 × (1-momentum)”更新教师。
            # _momentum_update 用学生参数的指数滑动平均刷新教师；教师不走 loss.backward 的梯度更新。
            self._momentum_update()
            # 再次出现的 image 仍是本批原始像素图；_m 的视觉编码器给出教师 token。
            image_embeds_m = self.visual_encoder_m(image) 
            # 教师图像 CLS 也映射到 [B,E]；torch.no_grad() 下不更新梯度。
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)
            # torch.cat([...],dim=1) 沿第 1 维（候选数）拼当前 B 条和历史 Q 条；
            # .detach() 切断旧队列的梯度图，最终候选矩阵为 [E,B+Q]。
            # .t() 把 [B,E] 转为 [E,B]；沿 dim=1 拼出当前 B 个和队列 Q 个候选。
            # 当前教师特征从 [B,E] 转成 [E,B]，与历史队列 [E,Q] 沿候选维拼为 [E,B+Q]。
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)
            
            # 教师文字编码器处理相同的文字，得到不参与梯度更新的稳定向量。
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            # text_output_m 是教师纯文本结果；取其 CLS 并投影成共同 E 维。
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            # 当前文本特征与历史队列拼成 [E,B+Q] 的候选集合。
            # _all 指当前批加历史队列，不是全数据集；每一列都是一个文字候选向量。
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            # @ 是矩阵乘法：[B,E] @ [E,B+Q] -> [B,B+Q]；
            # 温度 temp 用于调节 softmax 之后概率分布的尖锐程度。
            # i2t=image-to-text；每一行是当前图片，每一列是当前/历史文字候选；_m 表示教师目标。
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            # @ 是矩阵乘法；另一方向每行文字与全部图像候选比较。
            # t2i=text-to-image，方向与上一行相反；结果仍是 [B,B+Q]。
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp 

            # 构造和相似度矩阵同形状的硬标签，初始全部为 0。
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            # 当前 batch 中图 i 对应文字 i，故正确配对位于对角线。
            # 当前 batch 的第 i 张图配第 i 段文字，因此前 B 列中的对角线是硬标签正例；队列列保持 0。
            sim_targets.fill_diagonal_(1)

            # F.softmax(...,dim=1) 将每行候选分数变成和为 1 的概率；
            # 教师软分布与真配对的一热硬标签按 alpha 混合，缓解标注噪声。
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # alpha 越大，越相信教师的软概率；1-alpha 是真配对硬标签的权重。
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # 学生图像、文字向量分别与“当前教师 + 历史队列”计算相似度。
        sim_i2t = image_feat @ text_feat_all / self.temp
        # 此处 text_feat 仍是最初学生纯文本 CLS；image_feat_all 是当前教师+旧队列。
        sim_t2i = text_feat @ image_feat_all / self.temp
                             
        # F.log_softmax 得到每个候选的对数概率；逐项乘目标概率并求和取负，
        # 就是支持软标签的交叉熵；.mean() 再平均 B 个查询。
        # 每行先取 log_softmax，再乘目标概率、对候选求和；最后跨 B 个样本平均。
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        # 反向“句找图”沿同样的候选维算软标签交叉熵，再对 B 条句子取平均。
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        # 图找文、文找图两个方向的对比损失取平均。
        # ITA=Image-Text Alignment；两个检索方向的软目标交叉熵取平均，得到标量。
        loss_ita = (loss_i2t+loss_t2i)/2

        # 当前教师特征写入队列，供之后 batch 作额外候选。
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###============== Image-text Matching ===================###
        # clone() 保留原始 token ID，下面只修改用于图文融合的副本。
        # 再次出现的 text 是前面 tokenizer 返回的对象；clone 防止改 [ENC] 时覆盖原始 token ID。
        encoder_input_ids = text.input_ids.clone()
        # [ENC] 告诉 BERT：现在进行图文联合编码，而非纯文字编码。
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id
        
        # forward the positve image-text pair
        # 第 0 维是 batch；后面的负样本构造都按 bs 遍历。
        # bs=batch size，取 image 的第 0 维；这里用本 GPU 的批量数，区别于队列入队的全局 batch。
        bs = image.size(0)
        # 融合编码器让文字 token 读取图像 patch，计算真实配对表示。
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       # 交叉注意力的 K/V 来自图像 token。
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )            
        # 负样本抽样分布只用于选择下标，本身不需计算梯度。
        with torch.no_grad():       
            # 只在本 batch 的前 bs 个候选中抽负样本，不从历史队列抽图像张量。
            # 只取本批候选，因为历史队列没有完整图像 patch token；相似度经 softmax 变抽样权重。
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4 
            # 不允许抽到对应真配对；剩余概率偏向“相似但错误”的难负例。
            weights_t2i.fill_diagonal_(0)
            # 反方向的难负例权重：每张真图可以配哪些错误文字。
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4  
            weights_i2t.fill_diagonal_(0)   
            
        # select a negative image for each text
        image_embeds_neg = []    
        # b 是当前第 b 段真文字；为其抽一张相似但不配对的图片。
        # b 是本 batch 的样本序号；此处为每张图/每段文寻找一个错误配对。
        for b in range(bs):
            # torch.multinomial(权重,1) 按非负权重随机抽 1 个下标；相似但错误的
            # 图片通常权重较高，所以得到比完全随机抽样更“难”的负例。
            # b 是第 b 句文字；neg_idx 是为它抽到的错误图片在本批中的整数位置。
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        # list 中 bs 个 [P,Dv] 张量堆叠为 [B,P,Dv]。
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_ids_neg = []
        # 负文字的 token ID 与 0/1 mask 必须成对收集，避免 PAD 处理错位。
        text_atts_neg = []
        # b 是本 batch 的样本序号；此处为每张图/每段文寻找一个错误配对。
        for b in range(bs):
            # 对当前图像抽一段错误文字。
            # b 是第 b 张图片；neg_idx 是为它抽到的错误文字在本批中的整数位置。
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        # 文本 token ID 和注意力 mask 都按同样顺序堆成一个 batch。
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        # stack 新增 batch 维；B 个 [T] mask 堆为 [B,T]。
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        # 前 B 项是真文字配错图；后 B 项是错文字配真图。
        # 前 B 行保留真文字，后 B 行是错文字；稍后与 [错图,真图] 按位置组成 2B 对负例。
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        # mask 沿完全相同的顺序拼接，前 B 个对应真文字，后 B 个对应负文字。
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        # 与 text_ids_all 顺序对齐：前 B 张错图，后 B 张真图。
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        # 图像 token 没有 PAD，两段都沿用本批全 1 的 image_atts。
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        # 一次处理 2B 个错误配对，得到融合后的文字 token 表示。
        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                            

        # B 个正例 + 2B 个负例的融合 CLS 统一送进 ITM 头。
        # vl=vision-language；先 B 对正例再 2B 对负例的融合 CLS，形状 [3B,Dt]。
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        # [3B,Dt] -> [3B,2]，每对图文各有“不匹配/匹配”两个分数。
        vl_output = self.itm_head(vl_embeddings)

        # 前 B 个正例标签为 1；后 2B 个负例标签为 0。
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        # F.cross_entropy 输入 [3B,2] 原始 logits 和 [3B] 整数标签，
        # 内部做 log_softmax 与负对数似然，返回平均二分类 loss。
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        
        ##================= LM ========================##     
        # LM 再从 tokenizer 原始 ID 复制；不沿用 ITM 已改为 [ENC] 的 encoder_input_ids。
        decoder_input_ids = text.input_ids.clone()      
        # BOS 表示从句子开头开始生成；不要原地修改前面编码用的 text.input_ids。
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        # 把 padding 标签设为 -100，PyTorch 交叉熵会忽略这些位置。
        # masked_fill 返回目标标签副本；-100 是交叉熵忽略标志，不是模型要预测的词。
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        # 解码器在图像条件下预测词；labels 触发下一个词的语言模型损失。
        # 语言模型内部会按自回归目标对齐输入与下一词标签。
        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        # BertLMHeadModel 内部比较“当前位置输出”和“下一个词标签”。
        # decoder_output 来自 BertLMHeadModel，内部已将当前位置预测对齐下一个词标签。
        loss_lm = decoder_output.loss
        # 分别是全局对齐、融合匹配、图像描述生成三个训练目标。
        return loss_ita, loss_itm, loss_lm
 


    @torch.no_grad()    
    def copy_params(self):
        """训练开始前把学生参数完整复制给动量教师，并冻结教师梯度。"""
        # 每一对都是 [学生模块, 教师模块]。
        for model_pair in self.model_pairs:           
            # zip 让两个模块中对应位置的参数一一配对。
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # .data.copy_ 原地复制数值，不保留计算图。
                param_m.data.copy_(param.data)  # initialize
                # 教师通过滑动平均更新，所以无需优化器计算它的梯度。
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        """用学生参数的指数滑动平均更新教师参数。"""
        # 这里的 model_pairs 是 __init__ 中保存的四对 [学生,教师] 模块。
        for model_pair in self.model_pairs:           
            # param 是当前学生参数；param_m 是与它位置对应的教师参数。
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # momentum=0.995 意味着教师保留 99.5% 旧值，只接收 0.5% 学生新值。
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        """把当前批次教师特征写入循环队列，供后续 batch 作负候选。"""
        # gather keys before updating queue
        # 分布式训练时先收集各 GPU 的特征，使每张卡维护相同队列。
        image_feats = concat_all_gather(image_feat)
        # text_feat 是当前 GPU 的教师文字特征；复数 text_feats 是全部 GPU 拼接后的结果。
        text_feats = concat_all_gather(text_feat)

        # gather 后的 batch_size 是所有 GPU 的样本数之和。
        # image_feats 是 all_gather 后跨所有 GPU 的教师图像特征；这里的 batch_size 是全局批量。
        batch_size = image_feats.shape[0]

        # queue_ptr 是单元素张量，int() 把它转成切片索引。
        ptr = int(self.queue_ptr)
        # 原实现要求队列长度能整除全局 batch，以免跨队列末尾切片。
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # 队列形状是 [E,Q]，而新特征是 [B,E]，故先转置为 [E,B]。
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        # 图文队列同列必须代表同一原始配对；因此写入区间与 image_queue 相同。
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        # % 实现环绕：写到队列末尾后又回到第 0 列。
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        # 将新整数 ptr 写回单元素 buffer，下一次调用从这个列号继续。
        self.queue_ptr[0] = ptr 


def blip_pretrain(**kwargs):
    """给训练入口用的工厂函数：把关键字配置传给模型构造器。"""
    # **kwargs 在此把参数字典展开为 med_config=...、image_size=... 等命名参数。
    model = BLIP_Pretrain(**kwargs)
    # 返回模型对象，供外层 pretrain.py 搬设备和放进优化器。
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensor 是当前进程的局部批特征；每张 GPU 先为全部进程各分配一个接收容器。
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    # 全部 GPU 互相交换张量；这里明确不提供跨 GPU 的梯度传播。
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # 沿 batch 维合并，得到总形状 [全局B,E]。
    # dim=0 是 batch 维；进程顺序必须在图像与文字两次收集中保持一致。
    output = torch.cat(tensors_gather, dim=0)
    # 返回 all_gather 后沿 batch 维拼好的全局特征张量。
    return output     


# typing.List 仅用于参数/变量类型提示，不会自动执行张量维度检查。
from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    """递归遍历模块树，让兼容的编码器与解码器层共享同一权重对象。"""
    # 记录未成功找到对应解码器层的编码器参数路径。
    uninitialized_encoder_weights: List[str] = []
    # 两个对象类名不同才尝试日志提醒；原文件此分支的 logger 没有定义，
    # 若实际触发会 NameError。这是原仓库代码潜在缺陷，教学副本不更改逻辑。
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        # 每次递归都应收到一对 PyTorch 模块，而不是普通张量或字符串。
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        # 碰到带 weight 的叶子层，且路径不属于需要跳过的 attention 层时，共享参数。
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            # 赋的是同一个 Parameter 引用，不是复制一份数值。
            encoder_pointer.weight = decoder_pointer.weight
            # hasattr 检查是否有偏置；若有，就让两边也共享同一个 bias 对象。
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                # 与 weight 一样，这里共享同一个 bias 参数对象，而非仅复制一次数值。
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            # 当前叶子层的 weight/bias 已共享，递归不必再往下找子模块。
            return

        # _modules 保存当前模块的直接子模块，递归继续往下一层找。
        encoder_modules = encoder_pointer._modules
        # _modules 是 nn.Module 保存直接子层的有序映射；继续递归比较同名层。
        decoder_modules = decoder_pointer._modules
        # 若解码器当前节点有直接子层，才继续在对应的编码器节点递归匹配。
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            # 先记下全部编码器子模块，递归匹配成功后逐个移除。
            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            # 层序号偏移：当解码器多出交叉注意力层时，后续数字子层需要错位对齐。
            encoder_layer_pos = 0
            # name 是直接子层名，module 是子模块对象；这里主要使用 name 查两侧对应层。
            for name, module in decoder_modules.items():
                # ModuleList 的层名常是 '0'、'1' 等数字字符串。
                if name.isdigit():
                    # BERT 的层列表以 0、1、2... 命名，需处理两边层号可能的偏移。
                    encoder_name = str(int(name) + encoder_layer_pos)
                    # 解码器当前层仍用原数字名；编码器可能要加上偏移后的名字。
                    decoder_name = name
                    # 当两侧数字子层类型与数量错开时，后面调整 encoder_layer_pos 跳过解码器额外层。
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        # 解码器多出的交叉注意力层没有编码器对应项，跳过它。
                        encoder_layer_pos -= 1
                        continue
                # 解码器专有子层没有编码器对应项，直接跳过，不做权重共享。
                elif name not in encoder_modules:
                    continue
                # 给递归设上限，防止意外的循环模块引用造成无限递归。
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    # 普通同名子层可直接按 name 在两棵模块树中寻找。
                    decoder_name = encoder_name = name
                # 对名称对应的子模块再次执行同一匹配流程。
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                # 已匹配的路径从“未初始化列表”候选中移除。
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            # 剩余路径表示未匹配到解码器子层，累计供诊断；原函数不返回该列表。
            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    # 从整个 BERT 模块的根节点开始递归。
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  
