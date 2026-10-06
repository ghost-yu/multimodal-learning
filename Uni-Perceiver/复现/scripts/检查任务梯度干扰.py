# 使用二维二次损失解释论文式1的一阶近似；不是原模型100批梯度分析的复现。
import json  # JSON标准库，保存小例子的实际梯度、损失变化与近似值。
from pathlib import Path  # 标准路径库，根据本脚本位置找到项目目录。
import torch  # PyTorch实际自动微分，本例在CPU使用64位浮点，减小数值误差。


def quadratic(parameters, target):  # parameters=共享参数θ，target=任务希望参数接近的位置a。
    return 0.5 * (parameters - target).square().sum()  # L(θ)=1/2||θ-a||²，.square逐元素平方，sum汇总为标量。


def compare(target_j, name, step=0.1):  # target_j是提供更新的任务j目标，name为示例名称，step=固定单位方向步长λ。
    parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)  # 共享二维θ起点[0,0]，requires_grad开启求导。
    target_i = torch.tensor([-1., -2.], dtype=torch.float64)  # 被影响任务i，起点梯度为[1,2]。
    target_j = torch.tensor(target_j, dtype=torch.float64)  # 把Python列表转换成同精度张量。
    loss_i = quadratic(parameters, target_i)  # 当前任务i损失。
    loss_j = quadratic(parameters, target_j)  # 当前任务j损失。
    gradient_i = torch.autograd.grad(loss_i, parameters)[0]  # autograd.grad返回tuple，取θ梯度[2]；不写入parameters.grad。
    gradient_j = torch.autograd.grad(loss_j, parameters)[0]  # 第二任务梯度[2]，每个损失图分别求导。
    direction_j = gradient_j / gradient_j.norm()  # norm默认L2长度；单位方向，忽略原梯度大小，符合式1的归一化更新。
    direction_i = gradient_i / gradient_i.norm()  # 同一批任务i的单位自更新方向，仅为本例特定基准。
    updated_by_j = parameters.detach() - step * direction_j  # detach断开旧图；沿负梯度下降，不是沿正梯度移动。
    updated_by_i = parameters.detach() - step * direction_i  # 同一批自身下降后参数，用于比较改善尺度。
    exact_improvement = (loss_i.detach() - quadratic(updated_by_j, target_i)).item()  # Δ=L旧-L新，正值代表i损失下降。
    self_improvement = (loss_i.detach() - quadratic(updated_by_i, target_i)).item()  # Δ_i当前同批自更新改善。
    approximation = (step * torch.dot(direction_j, gradient_i)).item()  # 一阶Taylor：λ*(g_j/||g_j||)·g_i。
    cosine = torch.dot(direction_i, direction_j).item()  # 当前两批梯度余弦，范围[-1,1]，不冒充论文完整期望比值。
    curvature_term = 0.5 * step ** 2  # 二次函数Hessian=单位阵，单位更新方向的二阶损失增量为λ²/2。
    assert abs(exact_improvement - (approximation - curvature_term)) < 1e-12  # 本例Taylor加二阶项应精确一致。
    return {'case': name, 'gradient_i': gradient_i.tolist(), 'gradient_j': gradient_j.tolist(), 'cosine': cosine, 'first_order_improvement': approximation, 'exact_improvement': exact_improvement, 'curvature_term': curvature_term, 'same_batch_self_improvement': self_improvement, 'finite_step_ratio': exact_improvement / self_improvement}  # 保存精确与近似的区别。


results = [compare([-2., -1.], '方向协作'), compare([2., 1.], '方向冲突'), compare([-2., 1.], '一阶正交')]  # j梯度分别[2,1]、[-2,-1]、[2,-1]。
assert results[0]['exact_improvement'] > 0  # 协作方向让任务i改善。
assert results[1]['exact_improvement'] < 0  # 冲突方向让任务i变差。
assert abs(results[2]['first_order_improvement']) < 1e-12  # 正交时一阶影响为零。
assert results[2]['exact_improvement'] < 0  # 但有限步受曲率影响仍恶化，不能将一阶零解读为严格完全无影响。
root = Path(__file__).resolve().parents[2]  # 定位Uni-Perceiver目录。
output = root / '复现/outputs/任务梯度干扰检查.json'  # 记录文件路径。
output.parent.mkdir(parents=True, exist_ok=True)  # 自动建输出目录。
record = {'scope': '二维二次函数的式1说明；非原模型/非100批期望估计，不能重现表1', 'step': 0.1, 'dtype': 'float64', 'results': results, 'all_checks_passed': True}  # 显式记录证据范围。
output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')  # 保存UTF-8结果。
print(json.dumps(record, ensure_ascii=False, indent=2))  # 终端显示实际运行结果。
