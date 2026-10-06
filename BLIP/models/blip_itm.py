# models.med 是本项目的 MED 文件：BertConfig 读取网络配置；BertModel 可做文字／图文编码。
from models.med import BertConfig, BertModel
# BertTokenizer 把句子分成词表 ID；本文件实际通过 init_tokenizer 包装器创建，直接导入未使用。
from transformers import BertTokenizer

# torch 提供张量、设备与矩阵乘；nn 是 neural network 的可训练层命名空间。
import torch
from torch import nn
# F = functional，函数式运算；F.normalize 单位化向量，不新增可训练参数。
import torch.nn.functional as F

# 同仓库公共函数：创建 ViT、初始化分词器（含特殊 token）、装载预训练权重。
from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_ITM(nn.Module):
    # ITM = image-text matching，图文匹配；同一类也支持 ITC 全局相似度分支。
    # 这个包装模型返回得分，未自动遍历、过滤或保存大规模训练数据。
    def __init__(self,
                 med_config = 'configs/med_config.json',
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 embed_dim = 256,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        # 先初始化 Module 父类，后面的 self.xxx 网络层才能被登记到参数表。
        super().__init__()

        # visual_encoder 是视觉模型对象；vision_width 是输出 token 宽度 Dv，例如 Base=768。
        # image_size 默认 384，patch16 对应 24×24 个 patch，加 CLS 后 P=577。
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        # tokenizer 是可调用分词器对象，不是当前 batch 的 token 张量。
        self.tokenizer = init_tokenizer()
        # med_config 先是路径字符串，这次赋值后成为配置对象；同名变量的类型已改变。
        med_config = BertConfig.from_json_file(med_config)
        # 文字交叉注意力的 Key/Value 来自图片，需知道视觉特征的输入宽度。
        med_config.encoder_width = vision_width
        # 此处只构建编码器结构；权重由下方 blip_itm 工厂按 pretrained 参数加载。
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        # text_width 是文字隐藏宽度 Dt，从真实编码器配置读取。
        text_width = self.text_encoder.config.hidden_size

        # 两个投影层分别把 Dv／Dt 映射为共同维度 E=embed_dim，默认 256，供 ITC 点积。
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # ITM 不用 256 维全局对比向量，而用融合 CLS 的 Dt 维特征；2 表示不匹配／匹配两类。
        self.itm_head = nn.Linear(text_width, 2)


    def forward(self, image, caption, match_head='itm'):
        # 输入 image 是 float [B,3,H,W]；caption 是 B 条字符串；match_head 是分支名字。
        # image_embeds 的 embeds 指每个位置的 embedding，形状 [B,P,Dv]，仍有全部 patch。
        image_embeds = self.visual_encoder(image)
        # size()[:-1] 去掉最后的 Dv，得到 [B,P]；全 1 表示每个图像 token 都可读。
        # dtype=long 建整数 mask，.to(image.device) 保证 mask 与图片在同一设备。
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        # text 是 BatchEncoding 对象：input_ids 和 attention_mask 都是 [B,35]。
        # padding 补至 35，truncation 截长句，return_tensors='pt' 返回 PyTorch 张量；这里不同于预训练的 30。
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(image.device)


        if match_head=='itm':
            # 默认 MED 多模态模式：文字自注意力后，再以文字 Query 读取 image_embeds 的 Key/Value。
            # 本包装器未将首 token 显式替换为 [ENC]，这点与 BLIP_Pretrain.forward 的 ITM 段不同。
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                       return_dict = True,
                                      )
            # output 是模型输出对象；last_hidden_state 为 [B,35,Dt]。
            # [:,0,:] 取每个样本首位置的全部特征，分类头输出 [B,2] 原始 logits。
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])
            # 如需匹配概率，调用者沿最后一维 softmax 再取类别 1；本方法不自动判阈值或删除数据。
            return itm_output

        elif match_head=='itc':
            # ITC = image-text contrastive，mode='text' 只编码文字，未进行图文融合。
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                            return_dict = True, mode = 'text')
            # 隔了一段再看到 image_embeds：仍是 [B,P,Dv] 图像序列；取第 0 位后投影到 [B,E]。
            # normalize(dim=-1) 令每个全局向量长度为 1，因此后面的点积等于余弦相似度。
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

            # .t() 将 [B,E] 转成 [E,B]；@ 矩阵乘产生 [B,B]，第 i 行第 j 列是图 i 与文 j 的得分。
            # 这个分支比较全部 batch 组合，ITM 分支则逐行评估当前配对；返回形状不相同。
            sim = image_feat @ text_feat.t()
            return sim
        # 原实现未提供其他 match_head 的报错分支；传错字符串将隐式返回 None。


def blip_itm(pretrained='',**kwargs):
    # 小写工厂函数：**kwargs 收集 image_size/vit 等命名参数，再展开交给类构造器。
    model = BLIP_ITM(**kwargs)
    if pretrained:
        # 空字符串不加载权重；非空时 load_checkpoint 可读取本地路径或 URL。
        model,msg = load_checkpoint(model,pretrained)
        # msg.missing_keys 列出模型所需而 checkpoint 缺失的参数名；assert 要求没有缺失。
        assert(len(msg.missing_keys)==0)
    # 返回可调用的 nn.Module；调用 model(image,caption) 时由 Module 分派到 forward。
    return model

