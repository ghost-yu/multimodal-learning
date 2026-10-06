# json.load 读取图片路径/描述组成的标注列表；os.path.join 拼图片路径。
import json
import os
# random.choice 在一张图有多条描述时随机选一句。
import random

# Dataset 提供按索引取样的父类接口。
from torch.utils.data import Dataset

# PIL.Image 解码图片并统一为 RGB；ImageFile 控制截断图片容忍设置。
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# pre_caption 清洗描述、统一大小写并截断到指定词数。
from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    # 只覆盖预训练主路径；上面的 re_* 类用于图文检索微调/评测。
    def __init__(self, ann_file, transform, max_words=30):        
        # ann_file 是一个或多个 JSON 路径；transform 是图片增强可调用对象。
        self.ann = []
        # ann_file 可以是多个 JSON 标注文件；合并后统一按下标取样。
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):
        
        # index 是 DataLoader 给的数据集行号；ann 是该行字典。
        ann = self.ann[index]
        
        # 一张图可能有多条描述；训练时随机选一条，形成图文正配对。
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        # 强制 RGB，保证视觉编码器总能收到 3 个颜色通道。
        # JSON 中 image 的值应是可访问的本机图片路径；这里才真正打开图片。
        image = Image.open(ann['image']).convert('RGB')
        # transform 将 PIL 图片随机裁剪/增强并转成模型所需的 [3,H,W] 浮点张量。
        image = self.transform(image)
                
        # 单条样本返回 (图片张量,Python 字符串)；文字 token 化在 Pretrain.train 中进行。
        return image, caption
            

    
