# BLIP `pretrain.py`：变量来历、命名原因与库函数索引

先打开[pretrain.py 逐行详注版](../pretrain.py)，遇到一个名字忘了从哪来，就在本页搜索。详注版保留了原仓库 [pretrain.py](../pretrain.py) 的执行逻辑；我用 Python AST 比对过两者的可执行结构。ViT、BERT、三项损失的内部变量在 [BLIP 通用词典](术语与库函数.md) 和 [blip_pretrain.py](../models/blip_pretrain.py) 中继续解释。

## 1. 三层函数之间的数据怎样传

```text
直接运行文件
  └─ parser.parse_args()      → args：命令行参数对象
  └─ yaml.load(...)           → config：训练配置字典
  └─ main(args, config)
       ├─ create_dataset(...) → datasets：只含一个 Dataset 的列表
       ├─ create_sampler(...) → samplers：只含一个采样器的列表
       ├─ create_loader(...)[0] → data_loader：真正逐批产出的加载器
       ├─ blip_pretrain(...)  → model：BLIP_Pretrain 模型对象
       ├─ AdamW(...)          → optimizer：优化器对象
       └─ 每轮 train(model, data_loader, optimizer, epoch, device, config)
            ├─ 每批 image、caption
            ├─ model(image, caption, alpha) → 3 个损失张量
            ├─ loss.backward()、optimizer.step()
            └─ 返回 train_stats：只供写日志的字典
```

**关键区别**：`args.config` 是配置文件的**路径**，`config` 是读取该文件后得到的**内容**。`datasets` 是列表，`datasets[0]` 才是 Dataset；`data_loader` 是 DataLoader。`model` 是可运行的对象，`checkpoint` 是保存到磁盘的字典，两者不是同一种东西。

## 2. `train()` 中的变量生命周期

| 名字与命名原因 | 第一次从哪里来、是什么类型 | 隔较远再次出现时要想起什么 |
|---|---|---|
| `model`：英文“模型” | `main()` 中 `blip_pretrain(...)` 的返回值；多卡时被 DDP 包装。 | 循环里 `model(image, caption, alpha)` 调用的是同一个模型；这一步返回三个损失，不会自己执行 `optimizer.step()`。 |
| `data_loader`：data + loader，数据加载器 | `main()` 中 `create_loader(...)[0]`；可迭代对象。 | `len(data_loader)` 是本轮 batch 数；`data_loader.sampler` 是与它绑定的采样器；循环从它取得 `image, caption`。 |
| `optimizer`：优化器 | `main()` 中 `torch.optim.AdamW(...)` 创建。 | `zero_grad()` 清旧梯度；`param_groups[0]['lr']` 读当前学习率；`step()` 真正更新参数。三个动作仍针对同一个对象。 |
| `epoch`：轮次 | `main()` 的 `for epoch in range(...)` 给出的整数，从 0 开始。 | `set_epoch(epoch)` 用它控制采样顺序；`epoch==0` 时才按 batch 预热学习率；`alpha` 的升温公式也使用它。 |
| `device`：设备 | `torch.device(args.device)`，通常是 CUDA 设备对象。 | `image.to(device)` 将当前图片批搬到模型所在设备；`caption` 仍是字符串列表。 |
| `config`：configuration，配置 | 程序入口的 `yaml.load(...)` 返回 Python 字典。 | `config['alpha']`、`config['warmup_steps']` 等从同一个字典取值；本文件不会在循环内重新读 YAML。 |
| `metric_logger`：metric + logger，指标记录器 | `utils.MetricLogger(...)` 的实例。 | `add_meter` 注册指标；`log_every` 包装循环并打印；`update` 加本批值；轮末 `synchronize_between_processes` 汇总。它不会影响训练 loss。 |
| `header`：日志抬头 | `'Train Epoch: [{}]'.format(epoch)` 字符串。 | 传入 `log_every` 供打印，人眼识别是哪一轮；不喂给模型。 |
| `print_freq`：print frequency，打印频率 | 整数 50。 | `log_every(..., print_freq, ...)` 每隔约 50 个 batch 打印一次。 |
| `i`：短循环索引名 | `enumerate(...)` 产出的批次下标，每轮重新从 0 开始。 | 预热函数用它表示第 0 轮内的步数；`alpha` 用它估算本轮进度。`i` 不是 image 的缩写。 |
| `image`：图片 | DataLoader 拼好的一批浮点张量，通常 `[B,3,H,W]`。 | `.to(device)` 后仍叫 `image`，只是张量可能搬到了 GPU；随后作为模型的第一个实参。 |
| `caption`：图片文字描述 | DataLoader 产出的 B 条字符串。 | 此训练入口不做 tokenizer；它把列表传给模型，模型内部才转换 token ID。 |
| `alpha`：常用希腊字母，表示混合权重 | 每个 batch 根据 `config['alpha']`、`epoch`、`i` 计算出的 Python 浮点数。 | 传给模型后混合 ITA 的教师软目标与真配对硬目标；它不是可训练参数或学习率。 |
| `loss_ita` / `loss_itm` / `loss_lm` | 模型返回的三个标量 Tensor。代码称 ITA，论文称对应目标 ITC；另两项是图文匹配与语言建模。 | 相加为 `loss` 用于反向传播；调用 `.item()` 得到日志数字后，不可再拿该数字调用 `backward()`。 |
| `loss`：总误差 | 三个损失张量相加得到的新标量 Tensor。 | `loss.backward()` 从它出发求参数梯度；后续只在本 batch 有效，下个 batch 重新赋值。 |
| `k, meter` | 末尾字典推导式中每一对“指标名、统计器”。 | `meter.global_avg` 是整轮均值；`"{:.3f}".format(...)` 把数转为三位小数字符串。 |

## 3. `main()` 和命令行入口的变量生命周期

| 名字与命名原因 | 首次来源与类型 | 后面复用的位置 |
|---|---|---|
| `parser`：参数解析器 | `argparse.ArgumentParser()` 创建的对象。 | 多次 `add_argument` 注册选项，最后 `parse_args()`。 |
| `args`：arguments 的缩写 | `parser.parse_args()` 返回的 Namespace。 | `args.device` 选设备；`args.checkpoint` 指续训文件；`args.output_dir` 指保存目录。 |
| `config`：配置 | `yaml.load(open(args.config,'r'),...)` 读取 YAML 后生成的字典。 | 传给 `main()`、`train()`，也存入 checkpoint；`args.config` 只是路径。 |
| `seed`：随机种子 | `args.seed + utils.get_rank()` 得到整数。 | 连续传给 PyTorch、NumPy 和 Python `random` 三套随机数生成器。 |
| `datasets`：复数表明是列表 | `[create_dataset('pretrain',...)]`；里面目前仅 1 个 Dataset。 | `datasets[0]` 取得真实数据集；随后传给 `create_sampler` 与 `create_loader`。 |
| `num_tasks`：任务数 | `utils.get_world_size()` 得到分布式进程总数。 | 传给 `create_sampler` 的 `num_replicas`，决定数据被分成多少份。 |
| `global_rank`：全局进程编号 | `utils.get_rank()` 得到 0、1、2...。 | 传给 `create_sampler`，决定本进程拿哪份样本。 |
| `samplers`：采样器列表 | `create_sampler(datasets,...)` 返回。 | 和 `datasets` 一起传给 `create_loader`；本例只有一项。 |
| `data_loader`：数据加载器 | `create_loader(...)[0]`，第一个且唯一一个 DataLoader。 | 传给每轮 `train()`；对象不在每个 epoch 重新创建。 |
| `model`：模型 | `blip_pretrain(...)` 返回 BLIP_Pretrain 对象，随后 `.to(device)`。 | 多卡时变量 `model` 被改为 DDP 包装器；`train()` 接收包装后的对象。 |
| `model_without_ddp`：没有 DDP 包装的模型 | 初次赋值直接引用 `model`；DDP 包装后改为 `model.module`。 | 保存 checkpoint 时调用它的 `state_dict()`，避免层名前出现 `module.`。 |
| `optimizer`：优化器 | `torch.optim.AdamW(params=model.parameters(),...)`。 | 断点续训时恢复状态，训练时每批更新，保存时再次取 `state_dict()`。 |
| `start_epoch`：开始轮次 | 默认 0；若有断点则设为 `checkpoint['epoch']+1`。 | `range(start_epoch,max_epoch)` 从这里继续。 |
| `checkpoint`：检查点 | `torch.load(args.checkpoint,...)` 返回字典；只在提供路径时存在。 | 读出其中的模型、优化器、epoch；不是当前模型对象。 |
| `state_dict`：状态字典 | `checkpoint['model']`；键是层或 buffer 的名称。 | `model.load_state_dict(state_dict)` 把已保存数值写回当前模型。 |
| `start_time`：开始时间 | 进入轮次循环前 `time.time()` 记录的 Unix 时间戳。 | 循环结束用新的 `time.time() - start_time` 求总耗时。 |
| `epoch`：当前轮次 | `for epoch in range(...)` 每轮获得一个整数。 | 调度学习率、调用 `train()`、日志和 checkpoint 文件名都复用它。 |
| `train_stats`：training statistics | `train()` 返回的日志字典，值已变成三位小数字符串。 | 只在主进程的 `log_stats` 中写日志；不再参与梯度计算。 |
| `log_stats`：日志字典 | 给 `train_stats` 的键加 `train_` 前缀，再加 epoch。 | 用 `json.dumps` 转成字符串写入 `log.txt`。 |
| `save_obj`：待保存对象 | 一个包含模型、优化器、配置、epoch 的字典。 | `torch.save(save_obj, path)` 写成 `.pth` 文件。 |
| `f`：file 文件对象的常用短名 | `with open(...,'a') as f` 取得。 | 只在 `with` 块中调用 `f.write(...)`。 |
| `total_time` / `total_time_str` | 前者是秒数，后者是格式化字符串。 | 用于训练结束时打印；与每轮学习率无关。 |

## 4. 重要函数要看“定义在哪里”

| 调用 | 定义来源 | 本文件使用的结果 |
|---|---|---|
| `ArgumentParser/add_argument/parse_args` | Python 标准库 `argparse` | 定义和解析命令行参数。 |
| `yaml.load/yaml.dump` | `ruamel_yaml`，在此文件别名为 `yaml` | 读写 YAML 配置；`load` 的返回值是 `config`。 |
| `torch.device/manual_seed/load/save` | PyTorch 主包 | 建设备、设随机种子、读/写检查点。 |
| `torch.optim.AdamW` | PyTorch 优化器包 | 创建 `optimizer`；`zero_grad`、`step` 是这个对象的方法。 |
| `torch.nn.parallel.DistributedDataParallel` | PyTorch 分布式模型包装类 | 包装 `model`，反向传播时同步梯度。 |
| `dist.barrier` | `torch.distributed`，别名 `dist` | 已初始化进程组内的同步点。 |
| `MetricLogger/SmoothedValue/init_distributed_mode/get_rank/get_world_size` | 本仓库 [utils.py](../utils.py) | 日志统计和进程信息；不是 PyTorch 官方同名类。 |
| `warmup_lr_schedule/step_lr_schedule` | 同一 [utils.py](../utils.py)，但在入口单独 `from utils import` | 第 0 轮逐批预热、每轮开始时衰减学习率。 |
| `create_dataset/create_sampler/create_loader` | 本仓库 [data/__init__.py](../data/__init__.py) | 产生数据集、采样器、加载器。 |
| `blip_pretrain` | 本仓库 [models/blip_pretrain.py](../models/blip_pretrain.py) | 工厂函数返回真正的 `BLIP_Pretrain` 模型类实例。 |

### 每个 `import`：实际调用与常用函数分开看

这里的“常用”帮助你以后读其他文件；**常用不表示本文件真的调用了它**。可以在详注版顶部找到相同的逐项注释。

| 导入的名字 | 本文件实际调用 | 初学者还常会遇到什么 |
|---|---|---|
| `argparse` | `ArgumentParser()` 创建解析器，`add_argument()` 注册可输入的 `--选项`，`parse_args()` 生成 `args`。 | `type=int` 把字符串解析成整数；`action='store_true'` 表示出现该开关就是 True。 |
| `os` | `os.path.join(目录,文件名)` 组合输出路径。 | `os.path.exists(path)` 判断存在；`os.environ` 读取环境变量。本文件没用前者，项目 `utils.py` 使用后者。 |
| `ruamel_yaml as yaml` | `yaml.load(open(...),Loader=...)` 读配置；`yaml.dump(config,文件)` 写配置。 | YAML 是带缩进层级的配置文本；读出来的 `config` 是普通 Python 字典。`as yaml` 仅设置别名。 |
| `numpy as np` | `np.random.seed(seed)` 设置 NumPy 随机数起点。 | `np.array` 建数组、`np.mean` 求平均，但**模型训练的图片张量仍是 torch.Tensor**。 |
| `random` | `random.seed(seed)` 设置 Python 原生随机数起点。 | `random.choice(items)` 随机选一项，`random.shuffle(items)` 原地打乱；本文件没用这两个。 |
| `time` | 前后两次 `time.time()` 的差是秒数。 | `time.sleep(seconds)` 暂停程序，本文件没有调用。 |
| `datetime` | `datetime.timedelta(seconds=...)` 表示一段时长。 | `datetime.datetime.now()` 读当前日期时间；本文件没有调用。 |
| `json` | `json.dumps(log_stats)` 将字典变成 JSON **字符串**。 | `json.dump(obj,file)` 直接写文件；`json.load(file)` 从文件读出 Python 对象。本文件只用 `dumps`。 |
| `from pathlib import Path` | `Path(args.output_dir).mkdir(parents=True,exist_ok=True)` 创建目录。 | `Path.exists()` 查存在；`Path('a')/'b'` 拼路径。本文件仍用 `os.path.join` 拼文件名。 |
| `torch` | `device`、`manual_seed`、`optim.AdamW`、`load`、`save`、`nn.parallel.DistributedDataParallel`。 | `torch.Tensor` 是多维数字容器；`.to(device)` 搬设备，`.backward()` 求梯度。 |
| `torch.nn as nn` | **未通过 nn 别名调用**。 | 在模型文件常用 `nn.Module`、`nn.Linear`、`nn.Embedding`、`nn.LayerNorm`。此处 DDP 通过 `torch.nn.parallel` 调用。 |
| `torch.nn.functional as F` | **未通过 F 别名调用**。 | 在 `blip_pretrain.py` 常见 `F.normalize`、`F.softmax`、`F.cross_entropy`；不必在入口硬找 `F.xxx`。 |
| `torch.backends.cudnn as cudnn` | 设置 `cudnn.benchmark=True`，让 cuDNN 针对输入形状搜索较快算法。 | 严格复现涉及更多设置；单改 benchmark 不保证完全确定。 |
| `torch.distributed as dist` | `dist.barrier()` 等进程同步。 | `dist.all_reduce()` 汇总统计；`dist.all_gather()` 收集张量，在项目其他文件可见。 |
| `from torch.utils.data import DataLoader` | **这个直接导入未使用**；项目 `create_loader()` 内部创建 DataLoader。 | `DataLoader(dataset,batch_size,num_workers,sampler,drop_last)` 从 Dataset 批量取样。 |
| `from models.blip_pretrain import blip_pretrain` | `blip_pretrain(...)` 建并返回模型实例。 | 是项目函数，不是第三方库、也不是模型类名本身。 |
| `import utils` | `utils.MetricLogger`、`utils.SmoothedValue`、`utils.init_distributed_mode`、`utils.get_rank`、`utils.get_world_size`、`utils.is_main_process`。 | 这是仓库的 `utils.py`；阅读函数时可以直接打开该文件查定义。 |
| `from utils import warmup_lr_schedule, step_lr_schedule` | 第 0 轮按 batch 预热；每轮开始时按 epoch 衰减。 | 两个函数来自同一 `utils.py`；参数顺序分别是 `(optimizer,step,max_step,init_lr,max_lr)` 与 `(optimizer,epoch,init_lr,min_lr,decay_rate)`。 |
| `from data import create_dataset, create_sampler, create_loader` | 顺序建立 Dataset、DistributedSampler、DataLoader。 | 这些定义在仓库 `data/__init__.py`，不是 Python 标准库 `data`。 |

## 5. Python 小白先掌握这些表达式

| 源码写法 | 拆成最小步骤 |
|---|---|
| `if config['laion_path']:` | 从字典 `config` 取 `laion_path` 的值；空字符串视为 False，非空路径视为 True。代码并没有在这里检查目录真的存在。 |
| `len(data_loader)` | 调用 Python 内置 `len()`，返回加载器本轮会产生的 batch 数，不是图文总样本数。 |
| `min(1, progress)` | 两个数中选较小的；因此 alpha 的增长比例到 1 后不会继续涨。 |
| `for i, (image, caption) in enumerate(...)` | `enumerate` 每次产出“编号、一个 batch”；这个 batch 再解包成图片与文字。`i` 每个 epoch 从 0 开始。 |
| `[create_dataset(...)]` | 方括号创建只有一个元素的 Python 列表；`create_dataset(...)` 本身只返回一个 Dataset。 |
| `create_loader(...)[0]` | `create_loader` 先返回列表，`[0]` 再取其中第一个 DataLoader。 |
| `optimizer.param_groups[0]['lr']` | 从优化器的参数组列表取第 0 组，再从组字典中取键 `'lr'`。这段代码只有一个组。 |
| `range(start_epoch, max_epoch)` | 产生从起点到上界前一个整数，例如 `range(0,3)` 是 0、1、2。 |
| `{f'train_{k}': v for k,v in train_stats.items()}` | `.items()` 逐项产出键和值；给每个键加 `train_`，构成新字典。 |
| `{**some_dict, 'epoch': epoch}` | `**` 把已有字典的键值对展开，再添加当前轮次；不是乘方运算。 |
| `'checkpoint_%02d.pth' % epoch` | 旧式字符串格式化：整数不足两位时前面补 0，例如第 3 轮为 `checkpoint_03.pth`。 |
| `with open(path,'a') as f:` | `open` 以追加模式打开文件；缩进块内用 `f.write` 写入；离开块时自动关闭文件。 |
| `f.write(json.dumps(log_stats) + '\n')` | `json.dumps` 得到字符串；`'\n'` 是换行符，使每轮日志单独占一行。 |
| `args.seed + utils.get_rank()` | 将用户给的整数种子与当前进程编号相加；这是普通整数相加，并不是 PyTorch 专有语法。 |

## 6. 原版代码本身的几个限制

- `--evaluate` 被解析，但本文件没有读取 `args.evaluate`；它不会自动进入评估模式。
- `--distributed` 使用 `type=bool`，命令行字符串 `"False"` 依然可能转成 `True`。`utils.init_distributed_mode()` 最终还会查看进程环境变量。
- `dist.barrier()` 无条件调用；未初始化分布式进程组时，单进程运行到这里可能报错。
- 默认配置的 `train_file` 是作者机器上的 `/export/...` 路径；需要换成本机真实数据，模型初始化还会读取/下载预训练权重。
- 设置随机种子有助于复现，但不足以保证跨设备、跨版本或并行执行时得到完全相同的数值。

这些是**原版行为的说明**，详注版没有改动训练逻辑。[PyTorch 随机性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)、[DataLoader 与 DistributedSampler 文档](https://docs.pytorch.org/docs/stable/data.html)、[Tensor.to 的 `non_blocking` 说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html)可用来核对库函数的通用语义；具体版本运行仍要以项目环境为准。
