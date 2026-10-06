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
# 关闭 Pillow 对超大像素数的警告阈值；只改变图片解码限制，不改变模型结构。
Image.MAX_IMAGE_PIXELS = None

# pre_caption 清洗描述、统一大小写并截断到指定词数。
from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    # re 是 retrieval（图文检索）的缩写；本类给检索微调提供图片、句子和图片 ID。
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        # ann_file 是 JSON 文件路径列表；每个文件应装含 image/image_id/caption 的字典列表。
        self.ann = []
        for f in ann_file:
            # json.load 解析整个标注文件；+= 将多个文件中的行接成一个大列表。
            self.ann += json.load(open(f,'r'))
        # transform 是图片增强函数；image_root 是相对路径的共同根目录。
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        # img_ids 把原数据中的 image_id 映射成从 0 开始的连续整数编号。
        self.img_ids = {}   
        
        # n 是下一张“新图片”可用的连续编号；同一 image_id 的多条描述共用编号。
        n = 0
        for ann in self.ann:
            # ann 是当前 JSON 字典；img_id 不是当前列表行号。
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        # DataLoader 用这个返回值知道样本条数；这里按“标注行”而非去重图片计数。
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        # index 是标注行号，ann 是该行图文配对字典。
        ann = self.ann[index]
        
        # 将根目录和标注中的相对图片路径接起来，再读 RGB 图片。
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        # PIL 图片经 transform 变为可供视觉编码器读取的 [3,H,W] 浮点张量。
        image = self.transform(image)
        
        # pre_caption 清理并裁到 max_words；返回仍是 Python 字符串。
        caption = pre_caption(ann['caption'], self.max_words) 

        # 三项依次供训练：像素、描述、连续图片 ID；同一图片的多描述 ID 相同。
        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    # 检索评估把图片去重、文字单独列出；txt2img/img2txt 用于 Recall 正确性判断。
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        # 评估 JSON 是单个路径，不是训练类接收的路径列表。
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        # self.text 存全部候选文字；self.image 存每张图路径；两张映射表记录真配对。
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        # txt_id 是展平后连续文字编号，img_id 是当前图片在评估列表中的行号。
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            # 每张图可有多条正确描述，因此值为文字 ID 列表。
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                # i 是同一图片内描述的局部编号；txt_id 是全评估集的编号。
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        # 评估 Dataset 每次只取一张图片；文字列表由检索评估程序单独编码。
        return len(self.image)
    
    def __getitem__(self, index):    
        
        # index 是评估图片列表行号；同一索引也供 img2txt 查询正确描述。
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        # 返回图片张量及其整数编号，不返回文字；评估时会做全图对全句相似度。
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
            

    
