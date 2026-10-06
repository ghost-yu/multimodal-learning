# 本脚本运行从原文件提取的 CaptioningHandler.eval_batch、BeamHypotheses 和 BertCaptioningLoss。
# 模型与分词器是固定输出的测试替身；不下载权重，不表示真实 BEiT-3 图片描述效果。
import ast  # ast=abstract syntax tree，抽象语法树；解析 Python，按类/方法名选择真实计算代码。
import json  # JSON 标准库，保存数值和原处理器返回结果。
from pathlib import Path  # pathlib 标准路径库，用 __file__ 找项目目录。
from types import SimpleNamespace  # 轻量属性容器，为 utils.BeamHypotheses 提供与原代码相同的访问形式。
import torch  # PyTorch，执行 CPU 张量运算、梯度计算和原 beam 代码。
import torch.nn as nn  # nn=neural network，损失类的父类 Module 和 KLDivLoss 从此获得。
import torch.nn.functional as F  # F=functional，原处理器用 F.log_softmax。

base = Path(__file__).resolve().parents[2]  # scripts→复现→beit3；定位相邻原始源文件。


def source_class(filename, class_name):  # 按类名提取AST类节点，不导入源文件顶部的大量可选依赖。
    tree = ast.parse((base / filename).read_text(encoding='utf-8'))  # UTF-8读取中文注释；ast解析不执行代码。
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)  # next取第一个匹配的类；缺失时报错而不伪造实现。


namespace = {'torch': torch, 'nn': nn, 'F': F}  # 被提取的原类所需的全局对象，保持原torch计算。
nodes = [source_class('utils.py', name) for name in ['BeamHypotheses', 'BertCaptioningLoss']]  # 提取两个原类全部方法。
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-utils-classes>', 'exec'), namespace)  # compile/exec执行已选原代码；仅本地可信文件。
handler_node = source_class('engine_for_finetuning.py', 'CaptioningHandler')  # 找原Caption处理器类。
eval_node = next(node for node in handler_node.body if isinstance(node, ast.FunctionDef) and node.name == 'eval_batch')  # 只运行评估方法，不触发tokenizer下载/初始化。
namespace['utils'] = SimpleNamespace(BeamHypotheses=namespace['BeamHypotheses'])  # 给原方法提供utils.BeamHypotheses类。
exec(compile(ast.Module(body=[eval_node], type_ignores=[]), '<original-eval-batch>', 'exec'), namespace)  # 编译真实方法体为函数；之后手动传self。


class FakeTokenizer:  # Fake=模拟替身，用8个词编号检验流程，没有真实SentencePiece模型。
    cls_token_id, pad_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3  # 特殊词编号仅为测试约定，不是BEiT-3真实词表编号。

    def batch_decode(self, ids, skip_special_tokens=True):  # 与原处理器调用的接口相同，保存进入decode前的IDs。
        self.last_ids = ids.tolist()  # Tensor.tolist将CPU张量转为嵌套列表，用于检查末尾MASK是否保留。
        words = {4: 'a', 5: 'dog', 6: 'cat', 7: 'runs'}  # 普通词测试映射。
        return [' '.join(words[token] for token in row if token in words) for row in self.last_ids]  # 过滤特殊词，拼成便于看懂的测试句。


class FakeModel:  # 固定按逻辑长度输出 a、dog、EOS，验证beam及缓存接口，完全没有视觉网络。
    def __init__(self):  # 初始化调用日志。
        self.calls = []  # 每轮记录图片是否传入、短输入和缓存长度。

    def __call__(self, image, text_ids, language_masked_pos, padding_mask, text_len, incremental_state):  # 形参匹配原处理器传参。
        count = text_ids.shape[0]  # count=BK，已包含所有图片的beam行。
        previous_length = 0 if not incremental_state else incremental_state[0]['prev_key'].shape[2]  # 看原处理器上轮裁掉MASK后的缓存长度。
        self.calls.append({'logical_text_len': text_len, 'image_present': image is not None, 'input_ids': text_ids.tolist(), 'previous_cache_length': previous_length})  # 记录真实原方法传来的信息。
        logits = torch.full((count, 2, 8), -9.0)  # [BK,2,V]，默认词分数很低；这不是训练出的logit。
        preferred = {2: 4, 3: 5}.get(text_len, 2)  # 初次偏好a，再dog，之后EOS。
        logits[:, 1, preferred] = 4.0  # 当前MASK位置最高分词。
        logits[:, 1, 6 if preferred != 6 else 7] = 1.0  # 添加较差替代项，让beam有候选竞争。
        cache_length = 5 if image is not None else previous_length + 2  # 首次模拟3个视觉token+CLS+MASK；之后添加本次2个短token。
        cache = torch.arange(count * cache_length, dtype=torch.float).view(count, 1, cache_length, 1)  # [BK,head=1,L,dim=1]，支持原index_select与裁切。
        return logits, {0: {'prev_key': cache, 'prev_value': cache.clone()}}  # 返回新缓存，模型替身不检查真实注意力数值。


def run_case(max_len):  # max_len决定是正常EOS结束，还是先碰到长度边界。
    tokenizer = FakeTokenizer()  # 每例独立保存最终IDs。
    handler = SimpleNamespace(tokenizer=tokenizer, num_beams=2, max_len=max_len, length_penalty=1.0, vocab_size=8, predictions=[])  # 模拟原self属性，避开真实初始化依赖。
    model = FakeModel()  # 每例独立记录调用过程。
    namespace['eval_batch'](handler, model, torch.zeros(2, 3, 2, 2), image_id=torch.tensor([10, 11]))  # 运行原方法；两张模拟图、两条beam，图片只是占位。
    assert model.calls[0]['image_present']  # 第一轮确实编码图片接口。
    assert all(not call['image_present'] for call in model.calls[1:])  # 后续只靠缓存，不重复传图。
    assert all(len(row) == 2 for call in model.calls for row in call['input_ids'])  # 每轮短文字输入恒为2，而逻辑历史增长。
    return {'max_len': max_len, 'predictions': handler.predictions, 'decoded_ids': tokenizer.last_ids, 'calls': model.calls}  # 输出原处理器结果和调用证据。


normal = run_case(8)  # 允许a dog后自然EOS。
boundary = run_case(4)  # 到dog时触发长度边界，检验保存前缀是否写入该词。
assert [item['caption'] for item in normal['predictions']] == ['a dog', 'a dog']  # 固定模型下正常流程符合预期。
assert [item['caption'] for item in boundary['predictions']] == ['a', 'a']  # 原长度分支未写入刚选中的dog，记录事实而不改原算法。
criterion = namespace['BertCaptioningLoss'](0.1, 0.0, 0)  # 运行原平滑KL损失，关闭丢弃高损失token。
logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]], requires_grad=True)  # 两个目标位置、3个词，开启梯度。
target = torch.tensor([0, 1])  # 每行正确词编号。
loss = criterion(logits, target, 1)  # 原类输出标量KL。
q = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])  # ε=.1，正确词.9，另两个各.05。
manual = (q * (q.log() - logits.log_softmax(1))).sum(1).mean()  # 按定义手算KL(q||p)，核对接口顺序。
assert torch.allclose(loss, manual)  # 与原模块应一致。
loss.backward()  # 实际求导，验证损失可反传。
assert torch.isfinite(logits.grad).all()  # 所有梯度有限，仍不代表真实训练稳定。
output = base / '复现/outputs/Caption流程检查.json'  # 本次检查结果路径。
output.parent.mkdir(parents=True, exist_ok=True)  # 目录不存在则创建。
record = {'scope': '原Handler/BeamHypotheses/损失类局部运行，模型和tokenizer为模拟，不是BEiT-3真实推理', 'torch_version': torch.__version__, 'normal_eos_case': normal, 'max_length_case': boundary, 'label_smoothed_kl': loss.item(), 'manual_kl': manual.item(), 'finite_gradients': True}  # 保存真实运行证据与局限。
output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')  # 保存UTF-8 JSON。
print(json.dumps({'normal_captions': normal['predictions'], 'boundary_captions': boundary['predictions'], 'kl': loss.item(), 'finite_gradients': True}, ensure_ascii=False, indent=2))  # 简短输出结果，详细调用记录在JSON。
