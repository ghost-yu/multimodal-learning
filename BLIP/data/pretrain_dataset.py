# 教学副本：下方保持原仓库可执行语句和已有注释，仅增加中文学习提示。
# 原文件：BLIP/data/pretrain_dataset.py。请配合 BLIP-从零重讲-04-任务与API.md 阅读。
# json 读取标注；os/glob 处理路径；random 是原仓库保留的导入。
# Python 标准库：json.load 把 JSON 文本读成 Python 列表和字典。
import json
# Python 标准库：os.path.join 把目录和文件名拼成适合系统的路径。
import os
# 原仓库保留的随机数模块导入；本数据集主读取路径没有直接调用 random。
import random

# 继承 Dataset 后实现 __len__ 和 __getitem__，DataLoader 才能按索引取数据。
# Dataset 规定 __len__ 和 __getitem__ 接口，使 DataLoader 可以按索引取一条样本。
from torch.utils.data import Dataset

# PIL 负责打开图片；ImageFile 允许读取有截断的图片文件。
# Pillow 图片类；Image.open 按路径读图，convert(RGB) 保证三通道。
from PIL import Image
# ImageFile 控制图片文件解码选项；下面允许不完整图片文件继续加载。
from PIL import ImageFile
# 两个全局设置放宽原始图片的读取限制。
# 全局放宽截断图片的读取；读坏图仍可能有质量问题，这行不会修复像素。
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 关闭 Pillow 的最大像素数限制；遇到极大图片时内存占用仍可能较高。
Image.MAX_IMAGE_PIXELS = None

# 本仓库文字清洗函数：接收原 caption 字符串和长度上限，返回清理后的字符串。
from data.utils import pre_caption
# 原文件第二次导入 os；glob 按 *.json 通配符找 LAION 分片路径。
import os,glob

class pretrain_dataset(Dataset):
    """一条样本返回 (图片张量, 清理后的 caption 字符串)。"""
    def __init__(self, ann_file, laion_path, transform): 

        # 基础图文标注先读入内存；每个条目通常含图片路径与 caption。
        # ann 是 annotation（标注）的缩写；此列表先收集所有基础图文标注字典。
        self.ann_pretrain = []
        # ann_file 是 JSON 文件路径列表，按顺序合并所有标注。
        # ann_file 是配置传来的 JSON 路径列表；f 是本轮处理的一个文件路径字符串。
        for f in ann_file:
            print('loading '+f)
            # json.load 把文件里的 JSON 数组变成 Python 列表；ann 是当前这个文件的条目。
            ann = json.load(open(f,'r'))
            # += 把当前文件中的样本条目追加到总列表。
            # 列表 += 列表表示追加条目，不是数字加法；总基础标注会逐文件增长。
            self.ann_pretrain += ann
        
        # 把构造参数保存为实例属性；空值表示本次只用基础标注。
        self.laion_path = laion_path
        # 只有配置真的提供 LAION 目录才读取分片；否则跳到基础标注分支。
        if self.laion_path:
            # 如果给出 LAION 目录，查找其中所有 JSON 分片。
            # os.path.join 拼目录和通配符；glob.glob 返回所有匹配 JSON 的路径列表。
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            # with 确保文件读完自动关闭；f 是文件对象而不是文件名。
            with open(self.laion_files[0],'r') as f:
                # ann_laion 是当前 LAION 分片的标注列表；与 ann_pretrain 拼成训练索引。
                self.ann_laion = json.load(f)  

            # 最终数据同时包含基础标注和当前 LAION 分片。
            # 列表 + 建新列表，前面基础条目，后面当前 LAION 条目；当前分片轮换后随之更新。
            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            # 未启用 LAION 时，最终可按索引访问的 annotation 直接指向基础列表。
            self.annotation = self.ann_pretrain
            
        # 图像变换由 data/__init__.py 的 create_dataset 传进来。
        self.transform = transform


    def reload_laion(self, epoch):
        # 按 epoch 轮换不同 LAION JSON 分片，避免一次把全部标注读入内存。
        # % 让 epoch 超过分片总数后从第一片重新轮换。
        # n 是选中的 LAION 分片序号，% 让 epoch 超过文件数后从头循环。
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        # 换 epoch 后读取编号 n 的另一个分片；n 来自 epoch 对文件数取余。
        with open(self.laion_files[n],'r') as f:
            # ann_laion 是当前 LAION 分片的标注列表；与 ann_pretrain 拼成训练索引。
            self.ann_laion = json.load(f)      
        
        # 列表 + 建新列表，前面基础条目，后面当前 LAION 条目；当前分片轮换后随之更新。
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        # DataLoader 用这个长度判断一个 epoch 有多少样本。
        return len(self.annotation)
    
    def __getitem__(self, index):
        
        # 按索引取得一个字典，通常包含 image 路径和 caption 文字。
        # index 是 DataLoader 请求的单条序号；ann 是这一条含 image/caption 键的字典。
        ann = self.annotation[index]
      
        # 统一成 3 通道图片并做与模型匹配的图像变换。
        # 字典键 image 指图片路径；Image.open 读文件；convert 强制三通道，避免灰度图尺寸不符。
        image = Image.open(ann['image']).convert('RGB')
        # 这里 image 从 PIL 图片变成经过裁剪、ToTensor、Normalize 的 [3,H,W] 张量。
        image = self.transform(image)
        # 这里只清理和截断文字，真正的 token ID 在模型 forward 中生成。
        # caption 来自当前 ann 的文字字段；pre_caption 清洗并限长，此时仍是 Python 字符串。
        caption = pre_caption(ann['caption'],30)
        
        # DataLoader 会把多条返回值合并成一批；文字仍保持 Python 字符串。
        return image, caption
