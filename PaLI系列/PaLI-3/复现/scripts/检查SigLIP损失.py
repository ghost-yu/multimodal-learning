# 这个脚本只核对 SigLIP 核心公式，不导入完整 Big Vision/JAX 训练器，也不训练 PaLI-3。
# 每行计算都保留中文说明；只用 Python 标准库，便于先理解损失。
import math  # math 是数学标准库；本脚本用 exp（指数）、log1p（准确计算 log(1+x)）、isclose（容差比较）。
import json  # json 是数据序列化标准库；dumps 把字典转换成 JSON 文本，保存实际数值证据。
from pathlib import Path  # pathlib 是路径标准库；Path 组织跨平台路径，避免手写大量反斜杠。


def negative_log_sigmoid(value):  # value 是带正负标签符号的分数；返回 -log(sigmoid(value))。
    return max(-value, 0.0) + math.log1p(math.exp(-abs(value)))  # 稳定的 softplus(-value)，指数参数始终非正，避免大正数溢出。


def loss_and_bias_gradient(scale, bias, batch_size=2):  # scale=正缩放，bias=共享偏置；batch_size 默认 2，与讲解例子对应。
    similarities = [[1.0 if i == j else 0.0 for j in range(batch_size)] for i in range(batch_size)]  # 单位阵模拟两塔已单位化且互相正交的嵌入点积。
    total_loss = 0.0  # total_loss=所有图文对损失之和；先累加，最后按图像数量 B 归一化。
    total_gradient = 0.0  # total_gradient=偏置 b 的梯度之和，说明损失确实为优化提供方向。
    logits = []  # logits=每张图片对全部文字的未转换概率分数矩阵，最后为 B 行 B 列。
    for i in range(batch_size):  # i 是图片行索引；range(B) 产生 0 到 B-1。
        row = []  # row 暂存当前图片对应的 B 个分数。
        for j in range(batch_size):  # j 是文字列索引；嵌套循环覆盖 B² 个图文组合。
            sign = 1.0 if i == j else -1.0  # sign=标签符号；对角是原配图文 +1，其余 -1。
            logit = scale * similarities[i][j] + bias  # s_ij=a*cosine+b；这里 similarities 是模拟点积，不是实际图片编码。
            row.append(logit)  # list.append 在当前列表尾部添加本格分数。
            signed_score = sign * logit  # 乘标签使正对希望分数大、负对希望原分数小。
            total_loss += negative_log_sigmoid(signed_score)  # += 表示加到累积值，遍历全部配对。
            total_gradient += -sign / (1.0 + math.exp(signed_score))  # d[-log sigmoid(y*s)]/db=-y*sigmoid(-y*s)，因 ds/db=1。
        logits.append(row)  # 当前行计算完毕，将整行加入分数矩阵。
    return total_loss / batch_size, total_gradient / batch_size, logits  # 返回 1/B 归约的损失、偏置梯度，以及分数矩阵。


loss, gradient, logits = loss_and_bias_gradient(2.0, -1.0)  # 数字例：a=2,b=-1，四格带符号分数都是 1。
expected_loss = 2.0 * math.log1p(math.exp(-1.0))  # 每行有两格，各损失 log(1+exp(-1))；行均值仍为两倍。
assert math.isclose(loss, expected_loss, rel_tol=1e-12)  # assert 核对预期损失；rel_tol 指相对误差容许值，此例非常严格。
step = 1e-5  # step 是有限差分的小扰动 h，与训练学习率不是一回事。
upper = loss_and_bias_gradient(2.0, -1.0 + step)[0]  # [0] 只取偏置增加 h 时的损失。
lower = loss_and_bias_gradient(2.0, -1.0 - step)[0]  # 只取偏置减少 h 时的损失。
numeric_gradient = (upper - lower) / (2.0 * step)  # 中心差分估计 dL/db，核对解析求导式。
assert math.isclose(gradient, numeric_gradient, abs_tol=1e-9)  # 该对称点偏置梯度约零；用绝对容差，避免零附近相对误差问题。
off_loss, off_gradient, _ = loss_and_bias_gradient(2.0, 0.0)  # 再在 b=0 的非对称点检查；_ 表示此处不使用分数矩阵。
off_numeric = (loss_and_bias_gradient(2.0, step)[0] - loss_and_bias_gradient(2.0, -step)[0]) / (2.0 * step)  # 同样计算中心差分。
assert math.isclose(off_gradient, off_numeric, rel_tol=1e-8, abs_tol=1e-9)  # 非零梯度点也应一致，避免只在零点偶然通过。
output = Path(__file__).resolve().parents[1] / 'outputs' / 'SigLIP损失检查.json'  # __file__ 是当前脚本路径；parents[1] 到复现目录，再指定输出文件。
output.parent.mkdir(parents=True, exist_ok=True)  # 建输出目录；parents=True 连同上层一起建，exist_ok=True 允许目录已存在。
record = {'scope': '纯 Python 数值公式检查，不是原 JAX 训练器运行', 'logits': logits, 'loss': loss, 'expected_loss': expected_loss, 'normalization': '1/B', 'bias_gradient_at_minus_one': gradient, 'bias_gradient_at_zero': off_gradient, 'finite_difference_at_zero': off_numeric, 'checks_passed': True}  # 记录结果及证据边界。
output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')  # ensure_ascii=False 保留中文；indent=2 用两空格缩进，utf-8 避免中文乱码。
print(json.dumps(record, ensure_ascii=False, indent=2))  # 控制台显示同一份实际结果，不涉及 API 密钥或任何图片数据。
