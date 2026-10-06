# 教学副本：下方保持原仓库可执行语句和已有注释，仅增加中文学习提示。
# 原文件：BLIP/models/blip.py。请配合 BLIP-从零重讲-04-任务与API.md 阅读。
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
# 原项目关闭 Python 警告输出；第一次读代码可暂时忽略这两行。
# Python 警告模块；下面 filterwarnings 只控制提示显示，不会影响模型参数。
import warnings
warnings.filterwarnings("ignore")

# VisionTransformer 把图片变成 patch token；插值函数适配不同图片分辨率的权重。
from models.vit import VisionTransformer, interpolate_pos_embed
# med.py 是 BLIP 改造的 BERT，可做纯文本编码、图文融合、文字解码。
from models.med import BertConfig, BertModel, BertLMHeadModel
# Hugging Face 的分词器类；from_pretrained 载入英文 BERT 词表。
from transformers import BertTokenizer

# PyTorch 主模块：张量、设备、权重加载和拼接函数来自这里。
import torch
# nn 提供 Module 基类与神经网络层；F 则是无独立层对象的函数式 API。
from torch import nn
# F 是 functional 的缩写；本文件保留该导入，主路径主要用 nn 模型类。
import torch.nn.functional as F

# Python 文件/路径库；os.path.isfile 判断 checkpoint 字符串是否本地文件。
import os
# urlparse 拆链接的 scheme 等字段；scheme 为 http/https 时按远程地址加载。
from urllib.parse import urlparse
# timm 的下载缓存函数；返回本地缓存路径，随后 torch.load 读取权重。
from timm.models.hub import download_cached_file

class BLIP_Base(nn.Module):
    """通用特征提取器：根据 mode 返回图像、文字或图文融合的 token。"""
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
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
        # nn.Module 父类先初始化，后续子模块才会被 PyTorch 正确注册。
        super().__init__()
        
        # 返回视觉编码器和它的输出维度，例如 ViT-Base 的 768。
        # 再出现于 Decoder 构造器：vision_width 是维度整数，visual_encoder 才是 ViT 模型。
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        # tokenizer 负责文字到 token ID 的转换。
        # Decoder 也需要同一套 [DEC]/[ENC] 特殊词，才能处理图片描述文本。
        self.tokenizer = init_tokenizer()   
        # 读取 JSON 里的 BERT 结构配置，并指定图像 token 的维度。
        # 同名局部变量从配置路径字符串变为 BertConfig 对象；后续 med_config.encoder_width 才能赋值。
        med_config = BertConfig.from_json_file(med_config)
        # med_config 在上一行已由路径变成配置对象；encoder_width 告诉交叉注意力图像宽度 Dv。
        med_config.encoder_width = vision_width
        # 同一个文字编码器既能单读文字，也能用交叉注意力读取图像。
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        """mode 决定只提图像特征、只提文字特征，或提图文融合特征。"""
        
        # 提前拒绝拼错的 mode，避免后面静默返回 None。
        # mode 只准三种字符串；image 取图像 token，text 取文字 token，multimodal 让文字读图。
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        # caption 目前是字符串；tokenizer 把它变成 input_ids 与 attention_mask。
        # caption 是字符串；text 则是 tokenizer 输出对象，包含 input_ids 和 attention_mask 两个张量。
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        # mode 是调用者传的字符串；image 分支只返回全部图像 token [B,P,Dv]。
        if mode=='image':    
            # return image features
            # 输出所有图像 token：[B,P,Dv]，而非只取 CLS。
            # image 是一批像素图；image_embeds 是 ViT 输出的全部图像 token [B,P,Dv]。
            image_embeds = self.visual_encoder(image)             
            # 返回 ViT 的所有 token，不只 CLS；下游若要整图向量再自行取第 0 位。
            return image_embeds
        
        elif mode=='text':
            # return text features
            # mode='text' 不给图像特征，只让文字 token 相互注意。
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                            return_dict = True, mode = 'text')  
            # last_hidden_state 包含每个文字位置的最终特征 [B,T,Dt]。
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            # image 是一批像素图；image_embeds 是 ViT 输出的全部图像 token [B,P,Dv]。
            image_embeds = self.visual_encoder(image)    
            # 图片 token 全有效，故 [B,P] 注意力掩码全部是 1。
            # 图片 token 全有效；size()[:-1] 去掉 Dv 特征维得到 [B,P] mask。
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
            
            # 用 [ENC] 标记“现在做图文联合编码”。
            # 将每句首个 token 设为 [ENC] 的整数 ID；这是图文融合路径，与生成的 [DEC] 区分。
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            # 多模态分支用已改 [ENC] 的文字 ID；下面传完整 image_embeds 做交叉注意力。
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       # 文字通过交叉注意力读取图像 token。
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            # 返回融合后的全部文字 token [B,T,Dt]。
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    """图像描述模型：训练时计算生成 loss，推理时逐词生成 caption。"""
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        # 再出现于 Decoder 构造器：vision_width 是维度整数，visual_encoder 才是 ViT 模型。
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        # Decoder 也需要同一套 [DEC]/[ENC] 特殊词，才能处理图片描述文本。
        self.tokenizer = init_tokenizer()   
        # 同名局部变量从配置路径字符串变为 BertConfig 对象；后续 med_config.encoder_width 才能赋值。
        med_config = BertConfig.from_json_file(med_config)
        # med_config 在上一行已由路径变成配置对象；encoder_width 告诉交叉注意力图像宽度 Dv。
        med_config.encoder_width = vision_width
        # LMHeadModel 在 BERT 顶部加词表分类头，用于预测下一个词。
        self.text_decoder = BertLMHeadModel(config=med_config)
        
        # prompt 是固定句首；训练时不让它本身贡献损失。
        self.prompt = prompt
        # 减 1 是排除 tokenizer 自动加入的一个特殊结束 token。
        # prompt_length 是固定前缀占的 token 数，-1 去掉 tokenizer 自动附加的结尾 token。
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption):
        """有标准 caption 的训练路径；返回语言模型损失。"""
        
        # 先把图像转换成一串视觉 token，并为所有图像 token 建有效掩码。
        # image 是一批像素图；image_embeds 是 ViT 输出的全部图像 token [B,P,Dv]。
        image_embeds = self.visual_encoder(image) 
        # 图片 token 全有效；size()[:-1] 去掉 Dv 特征维得到 [B,P] mask。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        # 本 batch 按最长句 padding，但单句至多保留 40 个 token。
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        # 第 0 个 token 改为 [DEC]，提示解码器开始生成。
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        # padding 位置设为 -100，交叉熵会忽略这些标签。
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        # prompt 是固定提示，不要求模型把提示本身当作目标去学习。
        # 固定提示开头不计生成损失；-100 是交叉熵忽略标签，不是词 ID。
        decoder_targets[:,:self.prompt_length] = -100
     
        # labels 触发内部语言模型损失；decoder 在每一步还能看图像 token。
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        # 返回单个标量，由外层训练脚本执行 backward。
        loss_lm = decoder_output.loss
        
        # 训练路径返回单个下一词语言模型损失，外层调用 backward() 更新参数。
        return loss_lm
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        """没有标准 caption 的推理路径：从固定提示开始生成并返回字符串列表。"""
        # 图片只编码一次；后续逐词生成时重复使用这份视觉特征。
        # image 是一批像素图；image_embeds 是 ViT 输出的全部图像 token [B,P,Dv]。
        image_embeds = self.visual_encoder(image)

        # sample=False 代表束搜索；一张图要复制 num_beams 份图像 token 对应多个候选。
        if not sample:
            # beam search 同时维护 num_beams 条候选句，图像特征需复制同样次数。
            # 束搜索中一张图有 num_beams 个候选句，每条候选都需要一份相同图像条件。
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        # 每个视觉 token 都有效；mask 形状 [B,P] 或 beam 情况下 [B*num_beams,P]。
        # 图片 token 全有效；size()[:-1] 去掉 Dv 特征维得到 [B,P] mask。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        # generate() 内部会把这个字典传给解码器的每一步。
        # 这个字典通过 **model_kwargs 展开为 generate 的关键字参数，使每一步生成都能看图。
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        # 为 batch 中每张图片准备相同的英文句首提示。
        prompt = [self.prompt] * image.size(0)
        # 把提示字符串转换为整数 token，并放到图像所在设备。
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        # 第一个位置使用解码起始 token。
        input_ids[:,0] = self.tokenizer.bos_token_id
        # 去掉提示末尾的结束符，后续生成才能从提示后继续接词。
        # 移除提示末尾的结束标志，让解码器从固定提示之后继续生成。
        input_ids = input_ids[:, :-1]

        # sample=True 走核采样，按词概率随机抽；否则走确定性更强的束搜索。
        if sample:
            #nucleus sampling
            # top_p 核采样从累计概率覆盖 top_p 的候选词中抽样，输出可变化。
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  # 生成 [SEP] 时表示一句话结束。
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            # Beam Search 同时保留 num_beams 条高分候选，通常输出更稳定。
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        # outputs 仍是 token ID，需要逐条转回可读字符串。
        captions = []
        # outputs 是每张图生成的 token ID 序列集合；逐条调用 tokenizer.decode 转成句子。
        for output in outputs:
            # 跳过 [DEC]/[SEP]/[PAD] 等特殊 token。
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            # 解码结果包含固定提示前缀；返回给用户时只保留新生成的部分。
            # decode 得到的句子还含固定提示；切片只保留新生成的描述部分。
            captions.append(caption[len(self.prompt):])
        # captions 是已去固定 prompt 前缀的 Python 字符串列表，一张图对应一条。
        return captions
    

def blip_decoder(pretrained='',**kwargs):
    """构造图像描述模型，并可选加载已有权重。"""
    # **kwargs 把参数字典展开为构造器的命名参数；返回创建好的模型对象。
    model = BLIP_Decoder(**kwargs)
    # pretrained 是路径/URL 字符串；空字符串表示只建结构，不在这里装外部权重。
    if pretrained:
        # missing_keys 不为空表示模型层没有成功从检查点加载。
        # 左边拆包：载入后的 model 与 missing/unexpected keys 诊断信息 msg。
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    # 工厂函数返回已构造、可选已装载预训练权重的模型对象。
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    """构造通用特征提取模型，并可选加载预训练检查点。"""
    # 通用特征提取模型的工厂函数；Base 可返回图、文或融合 token。
    model = BLIP_Base(**kwargs)
    # pretrained 是路径/URL 字符串；空字符串表示只建结构，不在这里装外部权重。
    if pretrained:
        # 左边拆包：载入后的 model 与 missing/unexpected keys 诊断信息 msg。
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    # 工厂函数返回已构造、可选已装载预训练权重的模型对象。
    return model        

def init_tokenizer():
    """以英文 BERT tokenizer 为基础，增加 BLIP 自己的特殊 token。"""
    # 首次运行可能从 Hugging Face 下载 tokenizer 文件。
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # [DEC] 表示进入生成模式；[ENC] 表示进入图文联合编码模式。
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    # 把 [ENC] 的整数 ID 存成易读属性，供 forward() 替换首 token。
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    # 返回已加 [DEC]/[ENC] 的 tokenizer 实例，不返回整数 ID 列表。
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    """根据 base/large 构建视觉 Transformer，返回模型和输出维度。"""
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    # create_vit 中 base/large 选不同层数、头数和 token 宽度；此处比较的是字符串。
    if vit=='base':
        # Base: 16×16 patch、12 层、12 个注意力头、每 token 768 维。
        vision_width = 768
        # Base 视觉模型：图片按 16×16 patch 切块，12 层、12 头，每个 token 768 维。
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        # Large: 同样 16×16 patch，但 24 层、16 个头、每 token 1024 维。
        vision_width = 1024
        # Large 视觉模型：24 层、16 头，每个 token 1024 维，通常更耗显存。
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    # 返回值是二元组：真正的 ViT 模型和整数输出维度，调用端会用两个变量拆包。
    return visual_encoder, vision_width

def is_url(url_or_filename):
    """判断检查点来源是 http(s) 链接，还是本地文件路径。"""
    # url_or_filename 同时可能是网页链接或本地路径；parsed 保存拆出的协议等部分。
    parsed = urlparse(url_or_filename)
    # 布尔结果：协议为 http/https 才视作远程权重链接。
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    """读取权重，并在图片分辨率改变时适配位置 embedding。"""
    # 先判断来源；远程先下载缓存，本地已存在文件则直接读取。
    if is_url(url_or_filename):
        # 远程权重先缓存到本地，再用 CPU 读取以减轻 GPU 峰值占用。
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        # torch.load 把权重文件反序列化为 Python 对象；先放 CPU 降低 GPU 内存峰值。
        checkpoint = torch.load(cached_file, map_location='cpu') 
    # 若不是 URL 但确实是本地文件，直接 torch.load 读取。
    elif os.path.isfile(url_or_filename):        
        # 已存在的本地 .pth 文件直接读取。
        # 本地路径不经过下载；map_location=cpu 仍先加载到 CPU。
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    # model 键保存按层名称索引的参数字典。
    # checkpoint 是读出的外层字典；model 键才是层名到权重张量的映射。
    # checkpoint 是外层字典；model 键保存层名到参数张量的真正权重映射。
    state_dict = checkpoint['model']
    
    # 例如 224 图像换成 384 图像时，patch 网格尺寸改变，位置向量需插值。
    # 按当前图片网格尺寸重采样视觉位置向量，使旧权重的 patch 个数匹配新 ViT。
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    # 只有模型含动量教师 ViT 时，才同步重采样教师位置向量。
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        # 动量教师 ViT 位置向量也必须同步按目标图片尺寸插值。
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    # 输出头维度不匹配的权重不能直接加载，先从字典中删掉。
    # 遍历当前模型期望的参数名；同名但形状不同的旧权重不允许直接复制。
    for key in model.state_dict().keys():
        # 只比较旧检查点确实包含的同名参数；缺失层交给 strict=False 报告。
        if key in state_dict.keys():
            # 例如新任务分类头或不同分辨率造成权重形状不合；下行从待加载字典移除。
            if state_dict[key].shape!=model.state_dict()[key].shape:
                # 只删除待加载字典的这一项，不会删除当前模型已有的层。
                del state_dict[key]
    
    # strict=False 允许任务头缺失；msg 会记录未加载或多余的层名。
    # strict=False 允许缺少新任务头；msg 仍应检查重要主干是否大面积缺失。
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    # 返回模型与 load_state_dict 诊断，调用者可查看 missing_keys/unexpected_keys。
    return model,msg
    
