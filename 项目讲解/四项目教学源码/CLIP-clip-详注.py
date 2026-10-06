# 教学副本：CLIP/clip/clip.py。只添加注释；与原文件 AST 相同。
# hashlib 校验文件 SHA256；os 管路径；urllib 下载；warnings 发警告；version 比 PyTorch 版本。
# PIL.Image 是图片对象；torchvision.transforms 是图片预处理层；tqdm 显示下载进度。
# _MODELS 把人可读模型名映射到权重网址；_tokenizer 是 SimpleTokenizer 实例。
# hashlib 是 Python 标准库的“哈希”模块。下方 sha256(文件字节).hexdigest() 返回 64 位十六进制校验串；
# 它验证权重文件是否与 URL 中标出的摘要一致，不参与神经网络计算。
import hashlib
# os 是标准库的操作系统接口：path.join 拼路径，path.isfile 判断普通文件，makedirs 建目录。
import os
# urllib 是标准库网络接口。urllib.request.urlopen(url) 返回可读取字节的响应对象。
import urllib
# warnings.warn(文字) 输出不终止程序的提示，与 raise RuntimeError（终止当前操作）不同。
import warnings
# packaging.version.parse 把版本字符串解析成可按数字语义比较的对象，避免把 "1.10" 当作小于 "1.8"。
from packaging import version
# typing 的 Union/List 只给读者、编辑器标注允许的参数和返回类型；它们不自动检查运行时数据。
from typing import Union, List

# torch 是 PyTorch 主库：torch.load 读权重、torch.zeros 建张量、torch.jit 管 TorchScript。
import torch
# PIL.Image 的 Image 对象保存原始图片；convert("RGB") 统一为三通道。
from PIL import Image
# Compose 顺序执行变换；Resize 调尺寸；CenterCrop 取中心；ToTensor 把像素变成 [C,H,W]；
# Normalize 按通道计算 (像素-均值)/标准差。下面每个函数都会在 _transform 处再结合参数解释。
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# tqdm 包住迭代/进度对象，显示权重下载的字节进度，不改变下载内容。
from tqdm import tqdm

# 相对导入中的点号表示当前 clip 包。build_model 根据权重字典创建 CLIP 模型并装入参数。
from .model import build_model
# SimpleTokenizer 是项目内 BPE 分词器；as _Tokenizer 改本文件内名字，前导 _ 提醒主要内部使用。
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

# 不同 torchvision 版本提供插值枚举的方式不同；try 先试新版，ImportError 才退回旧版。
try:
    # 新版 torchvision 将插值算法放在枚举 InterpolationMode；BICUBIC 是双三次插值。
    from torchvision.transforms import InterpolationMode
    # BICUBIC 名字是统一入口，后面的 Resize 不必关心底层常量来自哪个版本。
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # 旧版没有该枚举时使用 PIL 的对应常量；try/except 只处理导入失败。
    BICUBIC = Image.BICUBIC


# torch.__version__ 是当前 PyTorch 版本字符串；这里只提醒版本过旧，不阻止继续导入。
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


# __all__ 规定 `from clip.clip import *` 默认公开哪些名字，不是模型层列表。
__all__ = ["available_models", "load", "tokenize"]
# _tokenizer 是整个文件共享的一份分词器实例，后续 tokenize() 使用它的 encoder 与 encode()。
_tokenizer = _Tokenizer()

# _MODELS 的键是人能输入的模型名，值是权重 URL；RN 是 ResNet，ViT 是 Vision Transformer。
# URL 里倒数第二段是文件 SHA256；这份字典保存地址，不表示已经下载权重。
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    # 函数名的前导 _ 表示内部辅助函数；url 是远程地址，root 是本机缓存目录，返回本地文件路径。
    # root 是缓存目录字符串；exist_ok=True 表示目录已有时不报错。
    os.makedirs(root, exist_ok=True)
    # basename 从 URL 路径取最后一段，如 ViT-B-32.pt；filename 是文件名，不含目录。
    filename = os.path.basename(url)

    # 将 URL 按斜杠分段，倒数第二段是预期 SHA256 摘要，不是文件名。
    expected_sha256 = url.split("/")[-2]
    # join 安全拼本地目录与文件名；download_target 是完整目标路径，不是下载数据本身。
    download_target = os.path.join(root, filename)

    # 已有同名路径但它不是普通文件（可能是目录），无法当权重文件覆盖，故直接报错。
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        # open(...,"rb").read() 读出文件字节；sha256 算摘要；hexdigest 转可比较的字符串。
        # 与 URL 摘要一致就复用缓存，避免重新下载。这里原源码没有 with，教学副本保留原行为。
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            # 校验成功直接返回缓存文件路径；本轮不发网络请求。
            return download_target
        else:
            # 同名文件已有但内容不符，向用户发警告后继续往下重新下载。
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 两个 with 同时管理网络响应和本地二进制文件，结束时会关闭资源。
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # Content-Length 是服务器声明的总字节数；tqdm 的 unit_scale/unit_divisor 负责把字节显示为 KiB 等。
        # loop 只是进度条对象，不是训练循环；ncols=80 指显示宽度。
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                # 每次读取最多 8192 字节；buffer 是本次下载的一小块字节数据。
                buffer = source.read(8192)
                if not buffer:
                    # 网络流读到末尾返回空字节串；break 退出 while。
                    break

                # output 是本地文件句柄；把刚收到的一块原始字节写进去。
                output.write(buffer)
                # 进度条增加本块的字节数量，与文件写入互不替代。
                loop.update(len(buffer))

    # 下载完成再做一次完整摘要验证；不一致说明文件损坏，不能把它当成可靠权重使用。
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    # 此时本地文件已存在且通过摘要检查；函数返回路径字符串，不返回文件字节。
    return download_target


def _convert_image_to_rgb(image):
    # image 是 PIL.Image；convert 返回新的 RGB 图片。灰度、RGBA 等输入都变为模型预期的 3 通道。
    return image.convert("RGB")


def _transform(n_px):
    # n_px 是视觉模型需要的图片边长，例如 224；函数返回预处理流水线，并不立刻处理图片。
    # Compose 依列表顺序执行所有图片变换；返回可调用的 preprocess 对象。
    return Compose([
        # Resize 把较短边调到 n_px，保持宽高比；BICUBIC 决定重采样像素的方法。
        Resize(n_px, interpolation=BICUBIC),
        # CenterCrop 从中心裁出 n_px×n_px，保证最终空间尺寸固定。
        CenterCrop(n_px),
        _convert_image_to_rgb,
        # ToTensor 变成 float 张量 [3,n_px,n_px]，普通 8 位像素通常映射到 [0,1]。
        ToTensor(),
        # 两个三元组分别是 RGB 的均值和标准差；必须与训练权重的预处理约定一致。
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    # dict.keys() 给键视图，list(...) 变成实际列表；返回的是可传给 load 的名字，不代表本机有权重。
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    # name=预设模型名或本地权重路径；device=CPU/GPU；jit=True 请求 TorchScript 形式。
    # 返回 (model, preprocess)：前者是模型，后者是处理 PIL 图片的可调用对象。
    # 内置名字（如 ViT-B/32）会下载权重；本地文件路径则直接读取。
    if name in _MODELS:
        # name 是内置名；download_root 若未给就使用用户目录下缓存。_download 返回实际本地路径。
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 若 name 已是本地文件，则无需网络下载；model_path 仍只是路径字符串。
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # rb 是二进制读取；opened_file 是文件句柄，model_path 仍是路径字符串。
    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            # torch.jit.load 尝试解开 TorchScript；eval() 切到推理模式（关闭 dropout 等随机训练行为）。
            # jit=False 时先载入 CPU，以便下面从 state_dict 建普通 nn.Module。
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            # 能读成 JIT 时，此处没有单独的权重字典；普通模型分支可从 model.state_dict() 提取。
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                # 用户虽请求 jit，但实际文件格式不支持，改为普通 PyTorch 模型路径。
                jit = False
            # 读取一般 PyTorch 参数字典；map_location='cpu' 避免检查点原设备不可用。
            state_dict = torch.load(opened_file, map_location="cpu")

    # 默认返回普通 PyTorch 模型，便于检查每层；同时返回匹配的图片预处理。
    if not jit:
        # state_dict 是参数名字到张量的映射；build_model 按参数形状恢复普通 PyTorch 模型。
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            # CPU 路径用 float32，许多 CPU 算子对 float16 的支持/性能有限。
            model.float()
        # visual.input_resolution 说明该模型需要的图像边长，以它创建匹配的预处理。
        return model, _transform(model.visual.input_resolution)

    # 以下是仅 jit=True 才会走的 TorchScript 图修补。JIT 图会把设备/精度写成常量，
    # 即使 model.to(device) 了，也可能仍引用原 CUDA 设备，所以要逐个改图里的常量节点。
    # patch the device names
    # trace 执行一次只含“建标量并搬设备”的小函数，将正确设备固化进可检查的计算图。
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    # findAllNodes 查图中的 prim::Constant；筛含 Device 的节点并取最后一个作为替换模板。
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        # TorchScript Node 的属性可能是字符串、整数等；kindOf 告诉我们对应 getter 的名字。
        sel = node.kindOf(key)
        # getattr(node, sel) 按函数名取 getter，接着用 key 读取；返回节点属性值。
        return getattr(node, sel)(key)

    def patch_device(module):
        # module 是当前遍历到的 JIT 子模块；不一定每个子模块都带可访问 graph。
        try:
            # 有些模块暴露 .graph，有些没有；hasattr 先检查，避免直接访问属性报错。
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            # 即使 hasattr 为真，某些 JIT 图属性读取仍可能报 RuntimeError，退回空列表。
            graphs = []

        if hasattr(module, "forward1"):
            # 部分 JIT 模块有额外的 forward1 图，也必须检查。
            graphs.append(module.forward1.graph)

        for graph in graphs:
            # graphs 是当前模块的一个或多个 JIT 计算图，逐图查被写死的设备常量。
            for node in graph.findAllNodes("prim::Constant"):
                # 只修改属性值是 cuda 字符串的设备常量，其他常量保持原样。
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    # copyAttributes 复制设备模板节点的属性，替换旧 CUDA 设备信息。
                    node.copyAttributes(device_node)

    # Module.apply 递归访问所有子模块；再显式处理两个编码入口的 JIT 图。
    model.apply(patch_device)
    # 除递归子模块外，两个绑定方法可能单独持有 graph，也显式修补。
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        # 与设备修补类似：trace 一个 float32 转换，抽取 aten::to 的 dtype 常量节点。
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        # 在 traced 图中找到 aten::to 的第二个输入（dtype 参数），其 node 可当 float32 模板。
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        # float_node 是 JIT 图节点，不是 Python 浮点数，也不是模型参数。
        float_node = float_input.node()

        def patch_float(module):
            # 与 patch_device 同样遍历 JIT graph；本函数把已有 half 转换常量改成 float32。
            try:
                # module 是递归访问到的某个子模块；未必带 graph。
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                # 有属性但读取失败时跳过该模块的普通 graph。
                graphs = []

            if hasattr(module, "forward1"):
                # 某些 JIT 模块另有 forward1 图，同样纳入待处理列表。
                graphs.append(module.forward1.graph)

            for graph in graphs:
                # aten::to 是张量转设备/类型的 JIT 运算；只查这些节点，不碰其他计算。
                for node in graph.findAllNodes("aten::to"):
                    # aten::to 在不同 JIT 版本里 dtype 可能位于第 2 或第 3 个参数；逐个检查。
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        # 值 5 在此 PyTorch JIT 图中代表 float16；替换为上面 float32 模板节点。
                        if _node_get(inputs[i].node(), "value") == 5:
                            # 将该输入节点的属性替换为 float_node，保留原图的其他节点结构。
                            inputs[i].node().copyAttributes(float_node)

        # apply 递归修补普通子模块，再处理编码方法里可能独立存在的图。
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        # 除修补图常量，还把当前模型的浮点参数改成 float32。
        model.float()

    # JIT 模型的 input_resolution 是标量张量，item() 取 Python 整数供 Resize/CenterCrop 使用。
    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    # 统一输入格式：一句字符串变成只有一项的列表；后面就可对每句循环。
    if isinstance(texts, str):
        # 列表形式让后续 enumerate 每次都处理“一句话”；B=1 时也返回二维张量。
        texts = [texts]

    # SOT/EOT 分别标记一句文字的开始/结束；中间才是真正的 BPE token。
    # encoder 是 token 字符串→整数 ID 的字典；SOT=start of text，EOT=end of text。
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    # 与 SOT 相对，EOT 是句末标志；CLIP.encode_text 后续会取其隐藏向量。
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # 每句话变成整数列表，稍后补齐到统一的 context_length=77。
    # 列表推导式逐句做 BPE 编码，并在两端加开始/结束特殊 ID。
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    # 较旧 PyTorch 版本下选 long，兼容当时的 index_select；两条分支都生成 [句子数,context_length]。
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        # 旧版本选择 64 位整数 long，保证其索引相关运算可接受。
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        # 较新版本使用 32 位整数 int，输出形状完全相同，空位仍为 0。
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # i=句子行号，tokens=该句尚未补齐的 ID 列表；先检查长度再写入 result。
    for i, tokens in enumerate(all_tokens):
        # enumerate 给句子行号 i 与该句整数 ID 列表 tokens；后面把列表填入 result 的第 i 行。
        if len(tokens) > context_length:
            if truncate:
                # 允许截断只保留前 context_length 个位置，随后把末位恢复为 EOT。
                # 列表切片只保留前 context_length 项；随后强制最后一项为结束符。
                tokens = tokens[:context_length]
                # 截断后仍必须保留 EOT，否则 encode_text() 找不到结束位置。
                tokens[-1] = eot_token
            else:
                # 不允许截断时，立即告知用户哪条输入过长；原句 texts[i] 用于诊断。
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        # 第 i 行前 len(tokens) 列写本句整数 ID；后面的 0 保持为 padding。
        result[i, :len(tokens)] = torch.tensor(tokens)

    # 返回整数 token 矩阵 [B,77]；还没有做 Embedding，也没有得到句向量。
    return result
