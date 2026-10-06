'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

# argparse 读取 --config/--output_dir 等命令行参数；不是神经网络层。
import argparse
# os 处理目录与文件路径；ruamel_yaml 解析项目 YAML 配置。
import os
import ruamel_yaml as yaml
# np 和 random 与 torch.manual_seed 一起固定随机种子。
import numpy as np
import random
# time/datetime 统计总训练耗时；json 将每轮指标写成文本。
import time
import datetime
import json
# Path.mkdir 创建输出目录；parents=True 允许同时建立不存在的上级目录。
from pathlib import Path

# torch 负责张量/设备/保存；nn/F 是项目中常见的层和函数式算子别名。
import torch
import torch.nn as nn
import torch.nn.functional as F
# DataLoader 是按 batch 读取 Dataset 的标准 PyTorch 接口。
from torch.utils.data import DataLoader
# cudnn.benchmark 允许为固定输入大小挑较快算法；dist 提供多进程同步。
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# ALBEF 是本地预训练模型；位置插值在读取分辨率不同的检查点时使用。
from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
# 本仓库 BERT tokenizer 把 Python 字符串变成整数词 ID 与有效位置 mask。
from models.tokenization_bert import BertTokenizer

# utils 管分布式初始化和日志；dataset/scheduler/optim 是本仓库自己的子包。
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train 是一个 epoch 的循环；model 是 ALBEF 或 DDP 包装，data_loader 每次给图像和文字。
    # optimizer 管参数更新，scheduler 管学习率，config 是 YAML 读出的字典。
    # train
    # train() 切训练模式，会启用 dropout 等训练行为，不表示开始迭代。
    model.train()  
    
    # metric_logger（指标记录器）把许多 batch 的数值汇总；delimiter 是打印时的分隔字符串。
    metric_logger = utils.MetricLogger(delimiter="  ")
    # add_meter 注册一项统计；SmoothedValue 用最近 window_size=50 次值平滑显示。
    # fmt 中 value 表示当前平滑值，:.4f 表示保留四位小数；lr 的学习率更小故保留六位。
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    # header 是进度条文字；epoch 来自 main 的外层循环，表示当前第几轮训练。
    header = 'Train Epoch: [{}]'.format(epoch)
    # 每 50 个 batch 打印一次；step_size=100 只控制第一轮学习率预热的步进频率。
    print_freq = 50   
    step_size = 100
    # warmup_steps 是配置里的“预热 epoch 数”，乘 step_size 后得到预热小步数。
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        # DistributedSampler 的 set_epoch 让各轮采用不同随机排列，同时各进程划分仍一致。
        data_loader.sampler.set_epoch(epoch)

    # i 是本 epoch 的第几个 batch；image 是像素张量，text 是字符串列表。
    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 清除上一批残留梯度；PyTorch 默认会累积 .grad。
        optimizer.zero_grad()
  
        # 将图片搬到 CUDA/CPU 设备；non_blocking 在合适的内存条件下允许异步传输。
        image = image.to(device,non_blocking=True) 

        # Dataset 给的是字符串；这里转成整数 token ID 与 attention mask。
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)
        
        # 第一轮逐渐增大 alpha，避免刚开始训练时过度相信尚不稳定的教师。
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        # forward 同时返回补词、图文对齐、图文匹配三个损失。
        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha = alpha)
            
        # 三个标量损失求和；顺序分别是遮词、全局图文对齐、融合匹配。
        loss = loss_mlm + loss_ita + loss_itm
          
        # backward 只计算梯度；step 才真正改模型参数。
        loss.backward()
        optimizer.step()    
        
        # .item() 把只有一个元素的标量张量变成 Python 数字，避免日志保留计算图。
        # loss_mlm / loss_ita / loss_itm 分别是遮词、图文对比、图文匹配的当前 batch 损失。
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        # param_groups 是优化器参数组列表；第 0 组的 lr 是此时实际学习率。
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            # 第一轮每隔 100 个 batch 更新一次预热调度；// 得到整数预热步号。
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    # 多卡时汇总各进程指标；single-process 运行时就是本进程的统计。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    # 遍历已注册的 meter：键名 k 是 lr/loss_mlm 等，值转为三位小数的字符串供日志保存。
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    
    
def main(args, config):
    # args 是命令行参数对象；config 是从 Pretrain.yaml 读出的配置字典。
    # 原代码默认 distributed=True，因此会初始化进程组并在 epoch 末 barrier。
    utils.init_distributed_mode(args)    
    
    # torch.device 把命令行里的 cuda/cpu 字符串变成 PyTorch 的设备对象。
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # rank 是分布式进程编号；各进程加不同偏移，以免随机采样完全相同。
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # start_epoch 是下一次要训练的轮号；若随后从断点恢复，会改为旧 epoch+1。
    start_epoch = 0
    # max_epoch 与 warmup_steps 分别来自 YAML 的调度器配置；原代码键名写作 schedular。
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    # 返回 pretrain_dataset；外层列表是 create_sampler/create_loader 的统一多数据集接口。
    datasets = [create_dataset('pretrain', config)]
    
    if args.distributed:
        # num_tasks=进程总数，global_rank=当前进程编号；samplers 为每个 Dataset 指定不同数据切片。
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)         
    else:
        # 单进程由 DataLoader 自己负责打乱，不需要 DistributedSampler。
        samplers = [None]

    # [0] 取刚建的唯一训练 DataLoader；batch_size=YAML 中每进程配置值。
    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    # 从名称/本地路径加载 BERT 词表；并非在这里加载整个 BERT 模型。
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    # 初始化图像 ViT、文字 BERT、对齐投影、动量教师与队列；init_deit 会取视觉初始化权重。
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    # 将模型参数和缓冲区移动到与 image/text_input 一致的设备。
    model = model.to(device)   
        
    arg_opt = utils.AttrDict(config['optimizer'])
    # 依据 config['optimizer'] 创建 AdamW 等优化器；arg_opt 允许属性式访问配置。
    optimizer = create_optimizer(arg_opt, model)
    # arg_sche 是 scheduler 参数字典的可点号访问包装；返回值中的第二项这里没有使用。
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    
    # checkpoint 是已有完整训练检查点路径；resume=True 还恢复优化器、计划和 epoch。
    if args.checkpoint:
        # torch.load 先在 CPU 读取文件；checkpoint 是含模型/优化器等多个字段的字典。
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            # resume=True 是真正续训：除了模型权重，还要恢复优化器动量和学习率进度。
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            # 仅拿旧权重作初始化时，新图像分辨率可能改变 patch 数，需重采样位置向量。
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
        # load_state_dict 根据参数名字把 state_dict 中的张量复制进当前模型。
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
    
    # DDP=DistributedDataParallel；保存时要取底层模块，避免参数名前多出 module.。
    model_without_ddp = model
    if args.distributed:
        # 每张 GPU 一个进程；device_ids 指当前进程使用的 GPU 编号。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()

    # epoch 是完整遍历一次 DataLoader 的轮次编号。
    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            # 预热由 train 的 batch 循环控制；后续按 epoch 更新正常学习率计划。
            lr_scheduler.step(epoch+warmup_steps)  
            
        # train_stats 是本轮训练指标字典，来自 train() 最后的汇总返回值。
        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config) 
        if utils.is_main_process():  
            # 主进程独自写日志和权重，避免多卡同时写同一文件。
            # 字典推导式给每个训练指标加 train_ 前缀，例如 loss_mlm → train_loss_mlm。
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            # save_obj 是续训所需的完整检查点；state_dict 只保存模型参数及缓冲区。
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            # 将模型、优化器、调度器、配置与 epoch 一起保存，便于续训。
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
            
            # open(...,"a") 追加，不覆盖以前轮次；f 是打开的文件句柄。
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # 所有进程完成本轮后再进入下一轮；此处要求已成功初始化分布式进程组。
        dist.barrier()  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

# 只有直接运行 Pretrain.py 才进入命令行入口；被 import 时不会开始训练。
if __name__ == '__main__':
    # ArgumentParser 收集命令行参数；add_argument 的 default 是没传该参数时的值。
    # --config 指 YAML，--checkpoint 指已有权重/断点，--output_dir 指本轮输出目录。
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    # 注意原实现用 type=bool：命令行里的非空字符串如 "False" 也会被 bool 判为 True。
    # 教学阅读先认识其语义；实际续训时须核实传参方式，避免误恢复优化器状态。
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    # world_size 是分布式总进程数；dist_url='env://' 从启动器环境变量取得通信地址。
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    # parse_args 真正读取本次运行的命令行，形成 args.config 等属性。
    args = parser.parse_args()

    # open(args.config,'r') 打开 YAML 文本；yaml.load 解析成 Python 配置字典。
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # 若输出目录及上级目录尚不存在就创建；exist_ok=True 避免重复创建报错。
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 把实际采用的配置另存一份，之后可核对模型与训练超参数。
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
