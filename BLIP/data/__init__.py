# Dataset 产生单样本；DataLoader 把多条样本拼成 batch。
import torch
from torch.utils.data import DataLoader
# torchvision 提供裁剪、翻转、转张量与标准化。
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.nocaps_dataset import nocaps_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.vqa_dataset import vqa_dataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    """按任务名创建数据集；预训练只使用下面的 pretrain 分支。"""
    
    # 三通道分别减均值、除标准差，让图像数值范围匹配视觉预训练权重。
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # Compose 按列表顺序执行训练图片变换，每次读取可能得到不同裁剪/颜色。
    transform_train = transforms.Compose([                        
            # 随机裁剪出一块图，再缩放到 config['image_size'] 的正方形。
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            # 以默认概率随机水平翻转。
            transforms.RandomHorizontalFlip(),
            # 从指定增强列表里抽两项，每项强度为 5。
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            # PIL 图片转为 [3,H,W] 浮点张量，通常缩放到 [0,1]。
            transforms.ToTensor(),
            normalize,
        ])        
    # 验证/测试不做随机增强，保证每次输入相同。
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='pretrain':
        # train_file 是 JSON 标注列表，laion_path 可选；这里传入训练变换。
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)              
        return dataset  
    
    elif dataset=='caption_coco':   
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='nocaps':   
        val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return val_dataset, test_dataset   
    
    elif dataset=='retrieval_coco':          
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset    
    
    elif dataset=='retrieval_flickr':          
        train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='vqa': 
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'], 
                                    train_files = config['train_files'], split='train') 
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset, test_dataset
    
    elif dataset=='nlvr': 
        train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'],'train')
        val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'val')
        test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'test')     
        return train_dataset, val_dataset, test_dataset   
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    """给每个数据集准备分布式采样器，使不同 GPU 处理不同样本。"""
    samplers = []
    # zip 把数据集与相应的“是否打乱”设置配成一对。
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    """将 Dataset 与采样器包装成可以逐批迭代的 DataLoader。"""
    loaders = []
    # 这些参数都是列表；zip 按相同位置配成每个任务的一组设置。
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            # 有 sampler 时由 sampler 决定样本顺序，DataLoader 不再单独 shuffle。
            shuffle = (sampler is None)
            # 丢掉最后不完整批次，便于分布式训练的固定 batch 逻辑。
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        # num_workers 指并行读图进程数；pin_memory 有助于拷贝到 GPU。
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

