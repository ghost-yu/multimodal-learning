# =============================================================================
# 版权声明：BSD-3-Clause 开源协议
# 允许商业使用、修改、分发，但必须保留版权声明、不得用作者名义背书产品
# =============================================================================
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# By Junnan Li

# 阅读约定：这是用户粘贴稿的增强版，对应原仓库 BLIP/pretrain.py。
# B 表示一批图文的条数；本文件负责“组织训练”，ViT/BERT/三项损失的细节
# 在 models/blip_pretrain.py。一个变量名下次出现时，先问它在哪里赋值、类型是什么、
# 形状是什么；隔得很远时，下面会重复提醒来源。
# 这是放在“项目讲解”目录里的教学副本；实际训练入口仍是 BLIP/pretrain.py。
# 单独在此目录运行可能找不到 models/data 模块。注释也不会自动提供数据或权重；
# 本次只核对源码结构，没有实际启动训练。


# =============================================================================
# 第一部分：导入所有依赖库（每个库都附带详细功能、常用方法、本代码用法说明）
# =============================================================================

# 1. argparse = argument parser（命令行参数解析器）
# 功能：读取终端执行脚本时传入的 --config、--output_dir 等命令行选项，把字符串转成Python变量
# 核心常用方法：
#   - ArgumentParser()：创建解析器对象
#   - add_argument()：添加一个可接收的参数，指定名称、默认值、类型、帮助信息
#   - parse_args()：解析实际传入的命令行参数，返回包含所有参数的对象
import argparse

# 2. os = operating system（操作系统接口库）
# 功能：和操作系统交互，核心用于路径拼接、文件/目录操作、读取环境变量
# 本代码常用函数：
#   - os.path.join(路径1, 路径2)：拼接路径，自动适配Windows/Linux的分隔符差异
#   - os.path.exists()：判断文件/目录是否存在；本文件没有调用它
import os

# 3. ruamel_yaml：YAML 格式文件处理库（YAML 是人易读的配置文件格式）
# 功能：把 YAML 配置文件读取成 Python 字典，或把 Python 字典写回 YAML 文件
# 此仓库使用 ruamel_yaml；写成 `as yaml` 后，下面用 yaml.load/yaml.dump 即可。
# `yaml` 是本文件的局部别名，不表示另一个变量或额外加载了一个库。
# 常用方法：
#   - yaml.load(文件, Loader=yaml.Loader)：加载YAML为Python字典
#   - yaml.dump(字典, 文件)：把字典写入YAML文件
import ruamel_yaml as yaml

# 4. numpy = Numerical Python（数值计算基础库）
# 功能：提供多维数组和数学函数。本文件只调用 `np.random.seed(seed)`；
# 图片与模型的张量计算由 PyTorch 完成，不经过 NumPy。
# 常用领域：数组运算、线性代数、随机数生成、数据预处理
import numpy as np

# 5. random：Python 标准库自带的随机数模块
# 功能：生成随机数、随机打乱列表、随机选择元素等
# 本代码中用于设置 Python 原生随机数的种子，有助于复现；
# 设种子仍不能保证不同硬件或算法下结果完全相同。
import random

# 6. time：Python 标准库的时间模块
# 功能：获取时间戳、计算程序运行时长、程序休眠等
# 本代码用 time.time() 获取当前时间戳，计算总训练耗时
import time

# 7. datetime：Python 标准库的日期时间处理模块
# 功能：处理日期、时间、时间差的格式化与运算
# 本代码用 datetime.timedelta() 把秒数转换成「时:分:秒」的易读格式
import datetime

# 8. json：Python 标准库的 JSON 格式处理模块
# 功能：实现 Python 字典 ↔ JSON 字符串 的互相转换
# 本代码用 json.dumps() 把训练日志字典转成 JSON 字符串写入文件
import json

# 9. pathlib：Python 标准库的面向对象路径工具；可与 os.path 配合使用。
# Path 是核心类，代表一个文件或目录路径，用面向对象方式操作路径更直观
# 本代码用 Path.mkdir() 创建输出目录，支持递归创建、已存在不报错
from pathlib import Path

# =============================================================================
# PyTorch 深度学习框架相关库
# =============================================================================

# 10. torch：PyTorch 主模块，深度学习框架核心
# 功能：提供张量（Tensor）运算、自动求导、GPU加速、分布式训练等基础能力
import torch

# 11. torch.nn：神经网络组件模块
# 功能：提供搭建神经网络的所有「层」和「容器」，比如线性层、卷积层、损失函数、Module基类
# 常用类：nn.Module（模型基类）、nn.Linear（线性层）、nn.Embedding（词向量）。
# 注意：本文件导入的 `nn` 别名并没有被实际调用；DDP 写成 `torch.nn.parallel...`。
import torch.nn as nn

# 12. torch.nn.functional：函数式神经网络接口
# 功能：提供函数形式的神经网络操作（激活函数、池化、损失函数等）
# 常用：F.softmax（分数变概率）、F.normalize（向量归一化）、
# F.cross_entropy（分类损失）。但这个训练入口没有调用 F；这些函数在模型文件里。
# `F` 与 `nn` 的区别不是“F 一定没有状态”：F 是函数集合，具体参数可能由调用者传入。
import torch.nn.functional as F

# 13. torch.backends.cudnn：cuDNN 加速库配置接口
# cuDNN 是 NVIDIA 专为深度学习设计的 GPU 加速库，专门优化卷积等运算
# 可通过配置 benchmark、deterministic 控制运算速度和结果确定性
import torch.backends.cudnn as cudnn

# 14. torch.distributed：分布式训练通信模块
# 功能：提供多进程间的通信、同步、集体运算，是分布式数据并行（DDP）的底层基础
# 本代码中用 dist.barrier() 实现所有进程的训练同步
import torch.distributed as dist

# 15. DataLoader：数据加载器类，来自 torch.utils.data 工具包
# 功能：批量加载数据集，支持多进程加速、随机打乱、自定义采样、自动批处理
# 是 PyTorch 训练的常用数据加载方式；但本文件不直接实例化 DataLoader，
# 而是调用下方导入的项目函数 create_loader()。所以这个直接导入未被使用。
from torch.utils.data import DataLoader

# =============================================================================
# 项目自定义模块导入
# =============================================================================

# 从 models/blip_pretrain.py 导入同名**工厂函数** `blip_pretrain(**kwargs)`；
# 它内部创建并返回 `BLIP_Pretrain` 类的实例。函数和类不要混淆。
# BLIP = Bootstrapping Language-Image Pre-training（引导式图文预训练模型）
# 返回的模型实现图文对比（论文写 ITC、代码变量写 ITA）、图文匹配 ITM、
# 语言建模 LM 三项目标。
from models.blip_pretrain import blip_pretrain

# 导入项目工具模块 utils，包含日志统计、分布式工具、学习率调度等通用函数
import utils
# 从 utils 导入两个学习率调度函数：
#   warmup_lr_schedule：训练初期的学习率预热调度
#   step_lr_schedule：训练过程的阶梯式学习率衰减调度
from utils import warmup_lr_schedule, step_lr_schedule
# 从 data 模块导入三个数据创建函数：
#   create_dataset：创建数据集对象
#   create_sampler：创建分布式数据采样器
#   create_loader：创建最终的数据加载器
from data import create_dataset, create_sampler, create_loader

# =============================================================================
# 读代码前先看：import 名字与“这个文件实际用到了什么”
# =============================================================================
# argparse：实际调用 ArgumentParser/add_argument/parse_args；解析 --config 等。
# os：实际调用 os.path.join；拼输出文件路径。
# ruamel_yaml as yaml：实际调用 yaml.load/yaml.dump；读写训练配置。
# numpy as np：实际只调用 np.random.seed；没有用 NumPy 算模型张量。
# random：实际只调用 random.seed；没有在本文件直接 random.choice。
# time：实际调用 time.time 两次；求开始和结束时间差。
# datetime：实际调用 datetime.timedelta；把秒变成人可读时长。
# json：实际调用 json.dumps；把日志字典序列化为一行 JSON 字符串。
# Path：实际调用 Path(...).mkdir；建立输出文件夹。
# torch：实际用 device、manual_seed、optim.AdamW、load/save、nn.parallel.DDP。
# torch.nn as nn：**导入了，但本文件未使用这个别名**。
# torch.nn.functional as F：**导入了，但本文件未调用 F.xxx**。
# cudnn：实际设置 cudnn.benchmark=True；由 PyTorch 决定底层算法搜索。
# dist：实际调用 dist.barrier()；跨进程同步。
# DataLoader：**直接导入未使用**；真正建立对象的是 data.create_loader。
# blip_pretrain：项目工厂函数，返回 BLIP_Pretrain 类实例。
# utils：项目模块，用到 MetricLogger、SmoothedValue、进程信息函数等。
# warmup_lr_schedule/step_lr_schedule：从同一个 utils.py 显式导入的函数。
# create_dataset/create_sampler/create_loader：从 data/__init__.py 显式导入。
# “导入了”与“本文件真的用到了”是两件事；遇到 import 不要硬找不存在的调用。

# =============================================================================
# 缩写与形状：后文遇到这些名字先回来看
# =============================================================================
# ITC = Image-Text Contrastive（论文里的图文对比目标）；代码变量名 loss_ita。
# ITA = Image-Text Alignment（代码里的图文对齐名字），与此处 ITC 对应。
# ITM = Image-Text Matching（图文配对二分类）。
# LM = Language Modeling（根据图像和已有文字预测下一个词）。
# ViT = Vision Transformer（视觉 Transformer）；DDP = Distributed Data Parallel。
# YAML 是层级配置文本；JSON 是另一种结构化文本；两者最终可读成 Python 字典。
# B = 本 GPU 的 batch 大小；image 常是 [B,3,224,224]；caption 是长度 B 的
# Python 字符串序列；模型 forward 返回三个标量 Tensor，不返回新的 caption。
# `.item()` 只把标量 Tensor 变成日志用的 Python 数字；loss.backward() 必须在
# 仍有计算图的 Tensor 上调用。`[0]` 是 Python 下标，表示第一个元素。


# =============================================================================
# 第二部分：单轮训练函数 train
# =============================================================================

def train(model, data_loader, optimizer, epoch, device, config):
    """
    【函数功能】完成一个 epoch（完整遍历一遍训练集）的训练
    【参数命名解释】
    - model：英文 model = 模型。来自 main() 中的 `blip_pretrain(...)`；多卡时会被
      DistributedDataParallel 包装。类型是可调用的 PyTorch 模型，不是模型文件路径。
    - data_loader：data（数据）+ loader（加载器）。来自 main() 的 create_loader(...)[0]；
      每次迭代返回 `(image, caption)`，其中 image 是一批图片张量、caption 是字符串列表。
    - optimizer：英文 optimizer = 优化器。main() 中创建的 AdamW 对象；其
      param_groups 持有学习率，step() 会根据模型参数中的 .grad 更新权重。
    - epoch：训练轮次的整数编号，来自 main() 的 `for epoch in range(...)`。
      编号从 0 开始，不是“已经完成的轮数”。
    - device：设备对象，如 torch.device('cuda')；由 main() 根据 --device 创建。
    - config：configuration（配置）的缩写，来自 YAML 的 Python 字典；
      `config['alpha']` 等写法是按键取值，不是模型内部学到的参数。
    返回值：一个日志字典，键如 'loss_ita'，值是格式化后的字符串，如 '2.314'。
    """

    # -------------------------- 训练模式切换 --------------------------
    # model.train()：将模型切换为【训练模式】
    # 训练模式 vs 评估模式（model.eval()）核心区别：
    # 1. Dropout层：训练模式随机失活部分神经元（防止过拟合）；评估模式全部启用
    # 2. 如果模型包含 BatchNorm，训练/评估时使用的统计方式也不同。
    # 这里调用 train() 是按 PyTorch 训练惯例设置所有子模块的状态；
    # 它不会自动执行反向传播，真正求梯度还要调用 loss.backward()。
    model.train()

    # -------------------------- 日志统计器初始化 --------------------------
    # metric_logger：指标记录器对象，命名 = metric（指标）+ logger（记录器）
    # 来自项目 utils.py，功能是统计训练过程的损失、学习率等数值，并做平滑格式化输出
    # delimiter="  "：多个指标打印时的分隔符，这里设为两个空格
    metric_logger = utils.MetricLogger(delimiter="  ")

    # add_meter()：向记录器注册一个需要统计的指标
    # 第一个参数是指标名称，第二个参数是统计器对象

    # 指标1：lr = learning rate（学习率），行业惯例用缩写，代表优化器当前学习率
    # SmoothedValue：滑动平滑统计器，命名 = Smoothed（平滑的）+ Value（数值）
    #   作用：保存最近 window_size 个 batch 和全局累计值。
    #   window_size=50：取最近50个批次的数值做平滑
    #   fmt='{value:.6f}'：显示**最新一个值**并保留6位小数；
    #   虽然窗口保存了50个值，这个格式并没有显示窗口平均值。
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    # 指标2：loss_ita = image-text alignment loss（代码里的图文对齐损失）；
    #   BLIP 论文将对应目标称作 ITC（image-text contrastive，图文对比）。
    # 对应 BLIP 的对比学习任务，衡量图像特征和文本特征的匹配程度
    # fmt='{value:.4f}' 只是日志显示4位小数，不改变实际 loss 张量精度。
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # 指标3：loss_itm = image-text matching loss（图文匹配损失）
    # 对应 BLIP 的图文匹配分类任务，判断图文是否为配对样本
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # 指标4：loss_lm = language modeling loss（语言建模损失）
    # 对应 BLIP 的文本生成任务，衡量模型生成图像描述的能力
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # -------------------------- 日志打印配置 --------------------------
    # header：日志头部字符串，打印日志时开头显示的标识，标明当前是第几轮训练
    # format(epoch)：把 epoch 变量填入 {} 占位符，例如 epoch=0 时显示 Train Epoch: [0]
    header = 'Train Epoch: [{}]'.format(epoch)
    # print_freq = print frequency（打印频率）
    # 每训练多少个 batch 打印一次日志，这里设为 50，即每50批输出一次当前指标
    print_freq = 50

    # -------------------------- LAION 数据集分片加载 --------------------------
    # config['laion_path']：配置字典中的 LAION 数据集路径（LAION 是大规模图文数据集）
    # 如果配置了 LAION 路径，说明使用分片式数据，每轮训练需要重新加载对应分片的标注
    # reload_laion(epoch)：数据集对象的方法，加载第 epoch 轮对应的标注文件
    if config['laion_path']:
        data_loader.dataset.reload_laion(epoch)

    # -------------------------- 分布式采样器设置轮次 --------------------------
    # data_loader 仍是本函数参数，来自 main() 的 create_loader(...)[0]；
    # .sampler 是它的 DistributedSampler 对象（sampler = 采样器）。
    # set_epoch(epoch)：分布式训练专用方法，设置当前轮次的随机种子
    # 作用：每轮训练打乱数据的随机顺序不同，避免所有 epoch 都按相同顺序训练，提升泛化性
    # 对 DistributedSampler，需在每轮迭代前调用以改变打乱顺序。
    data_loader.sampler.set_epoch(epoch)

    # -------------------------- 批次训练循环 --------------------------
    # enumerate()：Python 内置函数，同时返回元素的索引（批次编号）和元素本身
    # i：当前 epoch 内的批次编号（batch index）；每轮重新从 0 开始。
    # i 是代码作者沿用常见的短循环索引名，不是图片 image 的缩写。
    # (image, caption)：每一批数据，包含两部分：
    #   image：图像张量，来自 data_loader -> Dataset.__getitem__ -> DataLoader 拼批，
    #          形状通常 [本卡B, 3, H, W]；本文件 YAML 默认 H=W=224。
    #   caption：文本标注列表，每个元素是一个描述图片的字符串
    # metric_logger 是本函数上面创建的日志器；log_every 是生成器式包装器，
    # 原样逐批 yield data_loader 中的数据，同时每 print_freq 个 batch 打印日志。
    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # ---------------------- 学习率预热（仅第0轮） ----------------------
        # 仅第 0 个 epoch 执行学习率预热（warmup）
        # 预热原理：训练初期逐步增大学习率，减少一开始步长过大的风险；
        # 它不能保证“避免梯度爆炸”。
        # warmup_lr_schedule：自定义工具函数，命名 = warmup（预热）+ lr + schedule（调度）
        # 入参：优化器、当前批次号、预热总步数、预热起始学习率、初始学习率
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        # ---------------------- 梯度清零 ----------------------
        # optimizer.zero_grad()：把模型所有参数的梯度（.grad 属性）全部清零
        # 为什么必须清零：PyTorch 默认会【累加梯度】，即每次 backward() 都会把新梯度加到旧梯度上
        # 如果不清零，上一个 batch 的梯度会和当前 batch 混在一起，导致计算错误
        # 标准训练四步走：清零梯度 → 前向计算损失 → 反向传播梯度 → 更新参数
        optimizer.zero_grad()

        # ---------------------- 数据移到GPU ----------------------
        # .to(device)：PyTorch 张量方法，把张量移动到指定设备（CPU/GPU）
        # 模型和数据必须在同一个设备上才能运算，否则会报设备不匹配错误
        # non_blocking=True：请求尽可能异步地将 CPU 张量传到 GPU；
        # 能否有性能收益取决于设备、内存与后续同步。create_loader() 设置了 pin_memory=True。
        # 注意：caption 是字符串列表，不需要移动设备，由模型内部的 tokenizer 转成张量
        image = image.to(device,non_blocking=True)

        # ---------------------- alpha 权重逐步提升 ----------------------
        # alpha：希腊字母 α 常用于写“混合比例”。这里是本批临时计算出的 Python 浮点数，
        # 不是模型的可训练参数，也不是优化器的学习率。来源是 config['alpha']。
        # 前 2 个 epoch 逐步把 alpha 从 0 提升到配置值（ramp up = 逐步提升）
        # 原因：训练初期教师模型本身不稳定，太高的 alpha 会误导学生模型，所以慢慢增加权重
        # 计算公式：配置alpha × min(1, 当前总步数 / 前2轮总步数)
        # epoch*len(data_loader)+i：按“每轮批次数固定”估算的当前全局批次编号。
        # 2*len(data_loader)：前 2 个 epoch 的总 batch 数，作为提升的总时长
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader)))

        # ---------------------- 前向传播计算损失 ----------------------
        # 此处 model 还是 train() 的参数；多卡时是 DDP 包装后的 BLIP 模型。
        # `model(...)` 调用 PyTorch 的 __call__，再进入 BLIP_Pretrain.forward()。
        # 返回三个零维 Tensor（标量损失），赋给 loss_ita/loss_itm/loss_lm。
        # 三个损失对应 BLIP 的三个预训练任务，多任务联合训练
        loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)

        # 总损失 = 三个损失直接相加，优化器最终要最小化这个总损失
        # 此处代码没有再为三项 loss 配额外系数；alpha 只混合 ITA 的教师软目标。
        # 下面的 loss 是三个张量相加产生的新张量，仍连接着求梯度的计算图。
        loss = loss_ita + loss_itm + loss_lm

        # ---------------------- 反向传播 + 参数更新 ----------------------
        # loss.backward()：反向传播，PyTorch 自动求导的核心方法
        # 作用：从损失值出发，沿着计算图反向遍历，计算每个可学习参数的梯度，并存入 .grad 属性
        # 这一步只计算梯度，不会更新参数
        loss.backward()

        # optimizer.step()：优化器执行一步参数更新
        # 作用：读取每个参数的 .grad 梯度值，按照 AdamW 算法规则更新参数的数值
        # 执行完这一步，模型的权重才真正发生了变化
        optimizer.step()

        # ---------------------- 更新日志统计 ----------------------
        # .item()：PyTorch 张量方法，把只有一个元素的张量转换成 Python 普通数字
        # 为什么要用：当前日志器只接受 float/int；转成数字也不会把整张计算图
        # 长期留在日志器里。不能在 loss.backward() 前把训练用的 loss 替换成 .item()。
        # metric_logger.update()：更新对应指标的统计值，键是指标名，值是当前数值
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        # optimizer.param_groups[0]["lr"]：获取优化器第一组参数的当前学习率
        # param_groups：参数组，优化器可以把参数分成不同组设置不同学习率，这里只有一组
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # -------------------------- 进程间统计同步 --------------------------
    # metric_logger 仍是函数开头创建的同一个对象，里面存本轮各 batch 的记录。
    # synchronize_between_processes：进程间同步统计结果
    # 分布式训练时每个 GPU 只统计了自己的数据，需要汇总所有进程的指标，计算全局平均值
    metric_logger.synchronize_between_processes()
    # 打印本轮训练的全局平均统计结果
    print("Averaged stats:", metric_logger.global_avg())

    # 字典推导式生成结果字典：{指标名: 全局平均值（保留3位小数）}
    # k 是指标名称，meter 是对应的统计器对象，meter.global_avg 是全局平均值
    # 返回给 main() 的 train_stats 只用于写日志；字符串格式保留 3 位小数，
    # 已不参与训练与反向传播。
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# =============================================================================
# 第三部分：训练主入口函数 main
# =============================================================================

def main(args, config):
    """
    【函数功能】训练总流程：准备分布式环境、数据集、模型、优化器，然后循环调用 train 函数
    【参数命名解释】
    - args：arguments（命令行参数）的缩写。由底部 parser.parse_args() 创建，
      是 Namespace 对象；如 args.device、args.checkpoint、args.output_dir。
    - config：configuration（配置）的缩写。由底部 yaml.load() 从配置文件读出，
      是 Python 字典；如 config['image_size']、config['batch_size']。
    两者不同：args 管“怎样启动、输出到哪里”，config 管“训练用哪些超参数”。
    """

    # -------------------------- 初始化分布式环境 --------------------------
    # utils.init_distributed_mode：初始化分布式训练环境
    # 作用：设置当前进程的 GPU 编号、进程编号（rank）、建立进程间通信
    # 分布式训练：多个 GPU 进程各处理一部分数据；DDP 在反向传播时同步梯度。
    # 是否提速取决于通信开销、批量大小和硬件，不是调用此函数就一定更快。
    utils.init_distributed_mode(args)

    # -------------------------- 设置训练设备 --------------------------
    # torch.device()：创建设备对象，指定张量运算的硬件设备
    # args.device：命令行传入的设备类型，默认是 'cuda'（NVIDIA 显卡）
    # 注意：如果没有 CUDA 显卡，这个代码默认配置无法正常训练
    device = torch.device(args.device)

    # -------------------------- 设置随机种子（有助于复现） --------------------------
    # 设置本文件使用的几套随机数生成器的种子；这有助于复现，
    # 但 cuDNN、并行数据读取、CUDA 算法等因素仍可能使结果不完全相同。
    # seed：随机种子，随机数生成器的初始值
    # 每个进程的种子 = 基础种子 + rank，让各进程的随机数序列起点不同；
    # Dataset 样本如何分配主要由 DistributedSampler 决定。
    # utils.get_rank()：获取当前进程的编号（0 到 world_size-1）
    seed = args.seed + utils.get_rank()

    # torch.manual_seed()：设置 PyTorch 随机数种子；现代 PyTorch 也覆盖 CUDA
    # 设备的随机数生成器，不应简单理解为“只设置 CPU”。
    torch.manual_seed(seed)
    # np.random.seed()：设置 numpy 库的随机种子
    np.random.seed(seed)
    # random.seed()：设置 Python 标准库 random 的随机种子
    random.seed(seed)

    # -------------------------- 启用 cuDNN 自动调优 --------------------------
    # cudnn.benchmark = True：启用 cuDNN 自动搜索最快算法
    # 适用场景：输入图片形状固定时，能显著提升卷积运算速度
    # 注意：benchmark=True 使 cuDNN 为不同形状选择较快算法；
    # 关闭 benchmark 是严格复现的一个条件，但单独设 False 仍不保证完全确定。
    cudnn.benchmark = True

    # -------------------------- 创建数据集 --------------------------
    #### Dataset ####
    print("Creating dataset")
    # create_dataset：自定义函数，创建预训练数据集对象
    # 入参：
    #   'pretrain'：指定数据集类型为预训练
    #   config：传入配置字典，包含数据集路径、数据增广方式等
    #   min_scale=0.2：随机裁剪的最小比例，裁剪后图片面积至少是原图的 20%
    # create_dataset('pretrain',...) 本身返回一个 Dataset；外层 `[ ... ]`
    # 才把它包装为“只含 1 个数据集的列表”，以适配下面接收列表的辅助函数。
    datasets = [create_dataset('pretrain', config, min_scale=0.2)]
    # 打印训练样本总数，len(datasets[0]) 获取第一个数据集的样本数量
    print('number of training samples: %d'%len(datasets[0]))

    # -------------------------- 分布式采样器 --------------------------
    # num_tasks = number of tasks（总进程数）；常见训练方式是一进程配一 GPU，
    # 但这里的返回值严格说是通信进程数 world_size，不是机器台数。
    # utils.get_world_size()：获取分布式训练的总进程数
    num_tasks = utils.get_world_size()
    # global_rank = global（全局的）+ rank（编号），当前进程在所有进程中的编号
    # 0 号进程是主进程，负责写日志、保存模型等操作
    global_rank = utils.get_rank()

    # create_sampler：创建数据采样器（采样器决定每个进程每次取哪些样本）
    # 入参：
    #   datasets：数据集列表
    #   [True]：对应每个数据集是否是训练模式（训练模式需要打乱数据）
    #   num_tasks：总进程数，用于划分数据集
    #   global_rank：当前进程编号，决定取哪一部分数据
    # 每个 rank 取得自己的索引序列。注意 DistributedSampler 默认可能复制少量
    # 样本以补齐各进程长度，因此“绝对不重复”并不成立。
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)

    # -------------------------- 创建数据加载器 --------------------------
    # create_loader：创建 DataLoader 数据加载器列表
    # 入参：
    #   datasets：数据集列表
    #   samplers：采样器列表
    #   batch_size：每个 GPU 的批次大小，从配置读取
    #   num_workers：加载数据的子进程数，设为 4，多进程加速数据加载
    #   is_trains：[True] 表示此 DataLoader 属于训练数据，决定 shuffle/drop_last；
    #              它不等同于 model.train() 的模型状态切换。
    #   collate_fns：批处理整理函数，这里设为 None 使用默认方式
    # 返回值是列表，取 [0] 拿到第一个（也是唯一一个）预训练数据加载器
    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    # -------------------------- 创建模型 --------------------------
    #### Model ####
    print("Creating model")
    # blip_pretrain：从 models.blip_pretrain 导入的**工厂函数**，内部执行
    # BLIP_Pretrain(**kwargs) 并返回模型对象；赋值给变量 model。
    # 参数说明：
    #   image_size：输入图片的分辨率，比如 224 或 384
    #   vit：Vision Transformer（视觉 Transformer）的型号；本仓库接受字符串
    #        'base' 或 'large'，不是 'vit_base' / 'vit_large'。
    #   vit_grad_ckpt：是否启用 ViT 的梯度检查点，用计算时间换显存，降低显存占用
    #   vit_ckpt_layer：末尾多少个 ViT block 使用梯度检查点；
    #        create_vit 中的判断是 i >= depth - ckpt_layer，并非“第几层”。
    #   queue_size：历史图像/文字整体**特征向量**的列数 Q；
    #        队列不保存原始图片或句子字符串。
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

    # 把模型移动到指定 GPU 设备上
    # 模型的所有参数和缓冲区都会被移动到 device，后续运算都在该设备上执行
    model = model.to(device)

    # -------------------------- 创建优化器 --------------------------
    # AdamW：Adam with decoupled Weight decay，带“解耦权重衰减”的 Adam。
    # “解耦”指衰减直接作用于权重，而不是混进梯度统计；不应简单等同于 L2 正则项。
    # 入参：
    #   params=model.parameters()：要优化的参数，即模型所有可学习的权重
    #   lr=config['init_lr']：初始学习率，控制参数更新的步长
    #   weight_decay=config['weight_decay']：解耦权重衰减系数，可能帮助抑制过拟合。
    # 注意：model.parameters() 迭代结果也包含动量教师参数；但这些参数被设置
    # requires_grad=False、没有 .grad，所以优化器实际不会用梯度更新它们。
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    # -------------------------- 断点续训（可选） --------------------------
    # start_epoch = start（开始）+ epoch（轮次），起始训练轮次，默认从第 0 轮开始
    start_epoch = 0

    # args.checkpoint：命令行传入的断点文件路径，如果不为空说明要从断点继续训练
    if args.checkpoint:
        # torch.load()：加载 PyTorch 保存的检查点文件
        # map_location='cpu'：先把 checkpoint 张量读到 CPU，随后
        # load_state_dict() 会把数值复制进当前设备上的模型；这能避免
        # torch.load 直接将完整 checkpoint 映射到 GPU，但不保证显存一定充足。
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # state_dict = state（状态）+ dictionary（字典）。这里从 checkpoint
        # 取的是模型的 state_dict，包括参数与已注册 buffer（如 image_queue）。
        state_dict = checkpoint['model']
        # model.load_state_dict()：把参数字典加载到模型中，恢复模型权重
        model.load_state_dict(state_dict)

        # optimizer 仍是上面创建的 AdamW 对象；load_state_dict() 恢复它的
        # 一阶/二阶统计、步数、学习率等。此代码没有恢复随机数生成器状态，
        # 因此“完全恢复同一条随机训练轨迹”不能仅凭这些字段保证。
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 起始轮次 = 断点保存的轮次 + 1，接着上一轮继续训练
        start_epoch = checkpoint['epoch']+1
        print('resume checkpoint from %s'%args.checkpoint)

    # -------------------------- 分布式模型包装（DDP） --------------------------
    # model_without_ddp：变量名逐词是 model（模型）+ without（没有）+ DDP。
    # 这里首先让它引用上面已搬到 device 的原始模型对象，并没有复制新模型。
    # 因为 DDP 包装后，真实模型在 .module 属性里，保存模型时需要取原始模型，避免保存出 module. 前缀
    model_without_ddp = model

    # args.distributed：命令行参数，是否启用分布式训练
    if args.distributed:
        # DistributedDataParallel（DDP）：PyTorch 官方分布式数据并行工具
        # 作用：包装模型，自动在反向传播后同步所有进程的梯度，保证每个 GPU 模型更新一致
        # device_ids=[args.gpu]：指定当前进程使用的 GPU 编号
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # 上一行使 `model` 改为 DDP 包装器；原始模型并没有消失，位于
        # `model.module`。重新赋给 model_without_ddp，供后面保存时复用。
        model_without_ddp = model.module

    # -------------------------- 开始训练循环 --------------------------
    print("Start training")
    # start_time = start（开始）+ time（时间），训练开始时间戳
    # time.time()：返回当前时间的秒数（Unix 时间戳），用于计算总训练时长
    start_time = time.time()

    # range(start_epoch, config['max_epoch'])：生成从起始轮次到最大轮次的整数序列
    # 注意：range 是左闭右开区间，比如 range(0,20) 会生成 0~19，共 20 轮
    for epoch in range(start_epoch, config['max_epoch']):

        # step_lr_schedule：本仓库 utils.py 的函数。它每个 epoch 计算
        # max(min_lr, init_lr * decay_rate**epoch) 并写入优化器的 param_groups。
        # 函数名带 step，但这个入口每个 epoch 调用一次，而非每个 batch 调用。
        # 入参：优化器、当前轮次、初始学习率、最小学习率、衰减率
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])

        # 这些仍是 main() 前面创建的同一批对象：model（可能有 DDP 包装）、
        # data_loader（从 create_loader(...)[0] 取出）、optimizer（AdamW）、
        # device（torch.device）、config（YAML 字典）。只把 epoch 改成当前轮次。
        # train() 返回本轮日志字典，赋给 train_stats（training statistics）。
        train_stats = train(model, data_loader, optimizer, epoch, device, config)

        # -------------------------- 保存模型和日志（仅主进程） --------------------------
        # utils.is_main_process()：判断当前进程是不是主进程（rank=0）
        # 只让主进程写日志和保存模型文件，避免多进程同时写同一个文件导致冲突
        if utils.is_main_process():
            # train_stats 是本轮刚由 train() 返回的字典，如 {'loss_ita':'2.314'}；
            # `items()` 逐项给出键 k、值 v。这里的 k/v 是字典推导式局部变量，
            # 并非模型参数、图像或文字内容。
            # log_stats = log（日志）+ stats（统计），日志统计字典。
            # 把训练指标加上 train_ 前缀，再加上当前轮次，方便日志区分
            # 内层 `{f'train_{k}': v for ...}` 把键加 train_ 前缀；
            # 外层 `**` 把这个新字典展开，和 'epoch': epoch 合成最终字典。
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }
            # save_obj = save（保存）+ obj（对象），要保存的检查点对象
            # 保存四类信息，足够恢复模型和优化器继续训练；但不包含全部 RNG 状态。
            # 1. model：上文 model_without_ddp 引用的原始模型 state_dict，
            #    包括可训练参数与 register_buffer 注册的队列和指针。
            # 2. optimizer：优化器状态
            # 3. config：把本轮配置保存到文件，方便日后核对。
            #    原版续训代码并未自动比较 checkpoint['config'] 与当前 config。
            # 4. epoch：当前轮次编号
            save_obj = {
                # model_without_ddp 是前面保留的原始模型引用；即使本轮 train()
                # 使用了 DDP 包装的 model，这里仍从原始模型读取权重与 buffer。
                'model': model_without_ddp.state_dict(),
                # optimizer 是 main() 上方创建并在每个 batch 调用 step() 的同一对象。
                'optimizer': optimizer.state_dict(),
                # config 是程序底部读入的 YAML 字典，本轮训练中一直未重新创建。
                'config': config,
                'epoch': epoch,
            }
            # torch.save()：保存对象到文件
            # os.path.join()：拼接文件路径 = 输出目录 + 文件名
            # args.output_dir 是底部解析的 --output_dir 值，例如 output/Pretrain。
            # checkpoint_%02d.pth：旧式 % 格式化，%02d 表示最少两位数字补零，
            # 如 epoch=0 -> checkpoint_00.pth，epoch=12 -> checkpoint_12.pth。
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

            # f 是 file（文件对象）的常见短名，仅在此 with 缩进块中使用；
            # args.output_dir 仍是底部的命令行输出目录。追加模式 "a" 不清空旧日志。
            # json.dumps()：把 Python 字典转换成 JSON 字符串
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # -------------------------- 进程同步屏障 --------------------------
        # dist 是文件顶部 `import torch.distributed as dist` 的别名；
        # dist.barrier()：分布式屏障，barrier = 屏障/障碍。
        # 作用：阻塞所有进程，直到所有进程都执行到这一行，才继续往下走
        # 保证已初始化的所有进程到达这里才一起进入下一轮。
        # 原仓库无条件调用 barrier；若没有初始化进程组，单进程直接运行
        # 可能在这里报错。这是原版入口的运行限制，注释不改变其逻辑。
        dist.barrier()

    # -------------------------- 训练结束，打印总时长 --------------------------
    # start_time 是进入 epoch 循环前用 time.time() 记录的 Unix 时间戳；
    # 这里再次取时间并相减，得到 total_time（总训练耗时，单位秒）。
    total_time = time.time() - start_time
    # total_time_str：格式化的时间字符串
    # datetime.timedelta(seconds=...) 表示一段时长；int() 去掉小数秒，
    # str() 转成人可读形式，超过一天时可能显示“天, 时:分:秒”。
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# =============================================================================
# 第四部分：程序主入口
# =============================================================================

if __name__ == '__main__':
    # Python 标准主入口判断：直接运行本脚本时执行下面配置读取与 main；
    # 被其他模块 import 时不执行，避免意外启动训练。

    # -------------------------- 解析命令行参数 --------------------------
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()

    # --config：配置文件路径，默认读取当前目录下 configs/pretrain.yaml
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    # --output_dir：输出目录，保存检查点和日志，默认 output/Pretrain
    parser.add_argument('--output_dir', default='output/Pretrain')
    # --checkpoint：断点文件路径，用于续训，默认空表示从头训练
    parser.add_argument('--checkpoint', default='')
    # --evaluate：沿用了其他脚本的选项；**本预训练脚本并没有读取 args.evaluate**，
    # 因此加 --evaluate 不会自动跳过训练或开始评估。要判断选项是否生效，
    # 应在文件里搜索 `args.evaluate`，不能只看 add_argument 的名字。
    # action='store_true'：只要命令行加了 --evaluate，值就是 True，否则 False
    parser.add_argument('--evaluate', action='store_true')
    # --device：训练设备，默认 cuda
    parser.add_argument('--device', default='cuda')
    # --seed：随机种子，默认 42，type=int 指定类型是整数
    parser.add_argument('--seed', default=42, type=int)
    # --world_size：分布式总进程数，默认 1（单卡）
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    # --dist_url：分布式初始化的通信地址，默认从环境变量获取
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # --distributed：原仓库用 `type=bool` 解析命令行字符串；
    # Python 中 bool('False') 仍是 True，所以输入 --distributed False 并不能
    # 可靠关闭它。实际是否进入分布式还受 utils.init_distributed_mode() 检查
    # RANK/WORLD_SIZE 等环境变量的结果影响。这里保留原代码，仅说明行为。
    parser.add_argument('--distributed', default=True, type=bool)

    # 解析命令行参数，返回 args 对象，属性名就是参数名（去掉 --）
    args = parser.parse_args()

    # -------------------------- 读取 YAML 配置 --------------------------
    # config 的完整来路：args.config（文件路径） -> open(...,'r')（文件对象）
    # -> yaml.load(...)（解析内容） -> Python 字典 -> main(args, config)。
    # yaml.load()：加载 YAML 文件，转换成 Python 字典。
    # open(args.config, 'r')：以只读模式打开配置文件
    # Loader=yaml.Loader：指定加载器，ruamel_yaml 的标准用法
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # -------------------------- 创建输出目录 --------------------------
    # Path(args.output_dir)：创建 Path 对象，代表输出目录路径
    # .mkdir(parents=True, exist_ok=True)：创建目录
    #   parents=True：如果父目录不存在也一起创建（比如 output 不存在会先建 output）
    #   exist_ok=True：如果目录已经存在也不报错，直接跳过
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 把配置字典写回输出目录的 config.yaml，保存本次训练用的配置，方便后续复现
    # yaml.dump()：把 Python 字典转换成 YAML 格式字符串并写入文件
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # -------------------------- 启动训练 --------------------------
    # 最后再次提醒：这里的 args 是 parser.parse_args() 返回的 Namespace；
    # config 是 yaml.load() 返回的字典；main() 不会重新读取配置文件。
    # 所有准备工作完成，调用 main 函数开始正式训练。
    main(args, config)
