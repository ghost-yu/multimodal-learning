# 本脚本用本仓库的真实 CoCa 类搭建很小的网络，核对前向、标签、梯度和有限批次优化。
# 图片与文字是固定随机模拟数据，不能据此报告图片描述质量或原论文指标。
# 从本项目目录执行：& '复现/.venv/Scripts/python.exe' -X utf8 '复现/scripts/检查小型CoCa.py'

# json 是标准库，把检查记录保存成可再次读取的 JSON 文件，dump 是写入文件的函数。
import json
# sys 是标准库，path 保存 Python 搜索模块的目录列表；此处插入本仓库 src，确保使用本地代码。
import sys
# platform 是标准库，用 python_version() 记录 Python 版本，便于追溯运行环境。
import platform
# Path 是 pathlib 标准库的路径对象；比手工拼接 Windows 反斜杠更直观。
from pathlib import Path

# __file__ 是当前脚本的路径；resolve 取得绝对路径；parents[2] 从 scripts 向上两级到项目根目录。
project_dir = Path(__file__).resolve().parents[2]
# 把 src 放在搜索列表最前面，避免误导入系统中另一个同名 open_clip 安装。
sys.path.insert(0, str(project_dir / "src"))

# torch 是 PyTorch 主库；random/自动求导/优化器/张量都由它提供。
import torch
# CoCa 是本项目模型类；小配置缩小规模，但仍走真实模型实现的核心路径。
from open_clip.coca_model import CoCa
# CoCaTask 把模型输出与两个损失接起来，并构造右移标签。
from open_clip.task.coca_task import CoCaTask

# 设定随机种子，使本次 CPU 模拟实验可以重复；不表示所有 CUDA 环境完全逐位一致。
torch.manual_seed(7)
# 限制 CPU 线程数，避免极小矩阵使用太多线程产生调度开销。
torch.set_num_threads(2)

# vision_cfg=视觉配置；32 像素图像分成 8 像素 patch，共 4×4=16 个小块。
# width=隐藏宽度；head_width=每个注意力头宽度；32/8=4 个头；layers=Transformer 层数。
vision_cfg = dict(image_size=32, patch_size=8, width=32, head_width=8, layers=1,
                 attentional_pool="cascade", attn_pooler_queries=4,
                 attn_pooler_heads=4, output_tokens=True)
# text_cfg=文字配置；只定义 32 个人工编号，不能用原来的 49,408 词 CLIP tokenizer 来编码。
# context_length=最长输入长度；embed_cls=True 在末尾附加可学习整句汇总 token。
# pad_id=0、bos_id=1、eos_id=2 是这次人工词表的约定，不能照搬到真实词表。
text_cfg = dict(context_length=8, vocab_size=32, width=32, heads=4, layers=1,
                embed_cls=True, correct_cls_mask=True, output_tokens=True,
                pad_id=0, bos_id=1, eos_id=2)
# multimodal_cfg=多模态解码器配置；width 与输入的图文 token 宽度保持一致。
multimodal_cfg = dict(context_length=8, vocab_size=32, width=32, heads=4, layers=1)
# embed_dim 是图文全局对比向量的共同宽度 E；不是词表大小 V。
model = CoCa(embed_dim=32, vision_cfg=vision_cfg, text_cfg=text_cfg, multimodal_cfg=multimodal_cfg)
# device 是运算位置，本例明确使用 CPU，避免把能否使用 GPU 作为最小检查的条件。
task = CoCaTask(model, device=torch.device("cpu"), verbose=False)
# randn 从标准正态分布生成模拟图像 [B=2,C=3,H=32,W=32]，不是已经下载的真实照片。
images = torch.randn(2, 3, 32, 32)
# tensor 创建两个模拟文字序列；最后的 0 是人工 PAD，2 是人工 EOS。
texts = torch.tensor([[1, 3, 4, 5, 2, 0, 0, 0], [1, 6, 7, 8, 9, 2, 0, 0]], dtype=torch.long)
# ne=not equal，逐位置比较不等于 0，得到真实有效位掩码；它与 texts 同形状。
valid = texts.ne(0)
# batch 字典中的键对应模型 forward 的参数，Task 会用 **batch 展开这些参数。
batch = dict(image=images, text=texts, text_valid=valid)
# _caption_labels 是内部标签辅助方法；去掉 BOS，并把无效目标改为 -100。
labels = task._caption_labels(texts, valid)
# eval 关闭训练模式的 Dropout 等机制；no_grad 另外关闭求导记录，两者作用不同。
model.eval()
with torch.no_grad():
    # 普通模型前向返回 logits，而没有传监督 labels 给融合损失路径。
    outputs = model(**batch)
    # 融合路径用相同模型参数与标签，返回 caption CE 标量，不保存完整词表输出。
    fused_outputs = model(**batch, labels=labels)
    # Task 取出 [:-1] logits 并配上 [1:] labels，然后 CoCaLoss 计算普通路径 CE。
    plain_losses = task.loss(**task._build_loss_inputs(outputs, batch), output_dict=True)

# allclose 检查两种路径的 CE 是否在数值容差内接近；这是本小配置的检查，不是所有精度的证明。
ce_equal = bool(torch.allclose(plain_losses["caption_loss_ce"], fused_outputs["caption_loss_ce"], atol=1e-5, rtol=1e-5))
# AdamW 是带权重衰减的优化器；parameters() 收集网络中的可学习 Parameter。
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
# train() 开启训练模式；下面重复拟合同一个 batch，只演示模型能沿损失学习。
model.train()
# history 保存每一步总损失的 Python 数值，便于写报告；名字表示训练历史。
history = []
# range(20) 产生 0 到 19；这 20 步不是原论文的 500k 步，更不是 20 个 epoch。
for step in range(20):
    # zero_grad 清空上一轮梯度，set_to_none 避免给每个梯度张量写满零。
    optimizer.zero_grad(set_to_none=True)
    # training_forward 计算两个损失并汇总，尚未更新参数。
    losses, report = task.training_forward(batch)
    # finite 检查不是 NaN 或无穷；异常时立即终止，不能把失败结果写成通过。
    if not bool(torch.isfinite(losses["loss"])):
        raise RuntimeError("Non-finite loss")
    # backward 从总损失反向计算各参数梯度，保存到 Parameter.grad。
    losses["loss"].backward()
    # step 根据梯度与 AdamW 的状态更新权重，与循环变量 step 同名只是两个不同对象的成员/局部名字。
    optimizer.step()
    # detach 断开日志值的求导关系；item 将单元素 Tensor 转为 Python float。
    history.append(losses["loss"].detach().item())

# 遍历参数，确认至少有一个参数得到有限且非零的梯度；并不证明所有模块每一步都参与训练。
has_gradient = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0) for p in model.parameters())
# 检查真实 generate 接口，输出仍是这次人工词表编号，不能译成正常英文 caption。
# beam=候选续写；num_beams=2 只维护两个候选，seq_len=6 含起始词，最多总长 6。
generated = model.generate(images[:1], seq_len=6, min_seq_len=1, max_seq_len=8,
                           num_beams=2, num_beam_groups=1)
# 判断生成接口是否恢复原训练模式；generate 内部临时 eval，finally 应恢复 model.train 状态。
generation_restored_train_mode = model.training
# result 是记录字典，明确数据、配置、形状和检查边界，避免把模拟训练当成真实 caption 评价。
result = dict(python=platform.python_version(), torch=torch.__version__, device="cpu",
              data="fixed synthetic batch, no real image/caption evaluation", steps=20,
              image_features_shape=list(outputs["image_features"].shape),
              text_features_shape=list(outputs["text_features"].shape),
              logits_shape=list(outputs["logits"].shape), labels=labels.tolist(),
              fused_ce_matches_plain_ce=ce_equal, nonzero_finite_gradient=has_gradient,
              total_loss_history=history, loss_decreased=history[-1] < history[0],
              generated_token_ids=generated.tolist(), generation_restored_train_mode=generation_restored_train_mode)
# outputs_dir 是本项目复现记录目录；mkdir(parents=True) 自动建父目录，exist_ok 避免已有目录时报错。
outputs_dir = project_dir / "复现" / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
# encoding='utf-8' 保存中文；indent=2 使 JSON 缩进易读；ensure_ascii=False 保留正常汉字。
(outputs_dir / "小型CoCa检查.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
# 输出完整检查记录；里面没有 API 密钥、外部凭据或原论文性能声明。
print(json.dumps(result, ensure_ascii=False, indent=2))
# assert 不满足时抛 AssertionError，进程以失败退出，防止日志里出现结果却误判成功。
assert ce_equal and has_gradient and result["loss_decreased"] and generation_restored_train_mode, "Tiny CoCa check failed"
