# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer encoders both for text and for images."""

# 【中文解释】importlib 是 Python 标准库，负责按字符串动态导入模块。这里用 import_module("big_vision.models.vit") 选择编码器；返回模块对象，再访问模块里的 Model 类。常用 API 还有 reload(module)，但此处不重新加载。
import importlib
# 【中文解释】typing 是类型提示标准库：Any 表示任意类型；Optional[T] 表示 T 或 None；Tuple[A,B] 表示两项固定类型的元组；Union[A,B] 表示二选一。类型提示帮助阅读，不自动做运行时验证。
from typing import Any, Optional, Tuple, Union
# 【中文解释】absl 是 Abseil Python 工具库，logging 是日志模块；下方 logging.info(...) 记录加载信息。info/warning/error 分别表示普通信息、警告、错误；日志本身不加载参数。
from absl import logging

# 【中文解释】utils 是当前项目的工具模块，不是 Python 内置库。utils.load_params(checkpoint) 从检查点读取参数树；训练参数是嵌套结构，不等于一个图片张量。
from big_vision import utils
# 【中文解释】Flax 是基于 JAX 的神经网络库；linen 是模块接口，取别名 nn（neural network，神经网络）。这里使用 nn.Module、nn.compact；与 PyTorch 的 torch.nn 名字相似，但初始化/参数管理方式不同。
import flax.linen as nn
# 【中文解释】JAX 的 numpy 接口取别名 jnp（JAX NumPy）；提供 ones/exp/log/linalg.norm 等数组运算，可参与自动微分。它不是普通 numpy，也不是 torch；不要把几种库的张量任意混用。
import jax.numpy as jnp

# 【中文解释】ConfigDict 是 configuration dictionary（配置字典）的名字，本行只是 Any 的别名，没有定义新字典类。真正配置对象通常由上层传入。
ConfigDict = Any


# 【中文解释】Model 意为模型，继承 Flax 的 nn.Module。类字段是构造配置；Flax 通常 model.init(random_key, inputs) 建参数树，再 model.apply(variables, inputs) 计算，与 PyTorch 的直接调用方式有区别。
class Model(nn.Module):
  """Two towers transformer."""
  # 【中文解释】image 字段存图像编码器的配置，不是本次图片数据；__call__ 的同名 image 参数才是批量图片。None 表示使用默认配置，避免误把配置和数据混淆。
  image: Optional[ConfigDict] = None
  # 【中文解释】text 字段存文本编码器配置；Optional 表示允许 None。实际 token IDs 由调用方法参数 text 传入。
  text: Optional[ConfigDict] = None
  # 【中文解释】text_model 是文本模型模块路径字符串，不是已创建的模型。proj=image_text 项目目录；text_transformer 表示处理文字的 Transformer。
  text_model: str = "proj.image_text.text_transformer"
  # 【中文解释】image_model 默认模块 vit：ViT=Vision Transformer，视觉 Transformer。字符串只用来选择代码模块，具体规模由 image 配置决定，不能据此宣称加载了论文 2B ViT-G。
  image_model: str = "vit"
  # 【中文解释】out_dim=output dimension，输出特征维度。整数表示两塔共用维度；元组按图像、文字顺序。做点积匹配时两塔最后维度必须相等；128 是类默认值，不是 PaLI-3 的全部模型配置。
  out_dim: Union[int, Tuple[int, int]] = 128
  # 【中文解释】temperature_init 是温度相关参数的初值；本实现把 exp(t) 乘在相似度上，实际作为正的 logit scale（分数缩放系数），不是直接除以该值。初值必须大于零才能取 log。
  temperature_init: float = 1.0
  # 【中文解释】bias_init 是 bias（偏置）的初值；None 表示不创建偏置 b。SigLIP 训练器需要 extras["b"]，因此对应配置必须提供 bias_init；不能拿无偏置配置直接运行该损失。
  bias_init: Optional[float] = None

  # 【中文解释】@ 是装饰器语法：nn.compact 允许在此方法内部定义 Flax 子模块和参数；它告诉 Flax 如何组织参数作用域，并非把模型压缩为小模型。
  @nn.compact
  # 【中文解释】__call__ 使对象可像函数调用；self 是当前实例。image 通常 [B,H,W,3]，text 通常整数 IDs [B,L]；B=batch size，H/W=高宽，L=文本长度。**kw 接收额外关键字参数并转发，例如训练开关。
  def __call__(self, image, text=None, **kw):
    """Returns (B,C) image and (B,C) text representations."""

    # Support calling without text or without image, for example for few-shot.
    # 【中文解释】z 常表示 learned representation（学到的表示），txt=text，img=image；先置 None 是为了允许只计算一个塔。有效时两者分别为 [B,C]，C 为共享嵌入维度。
    ztxt, zimg = None, None
    # 【中文解释】out=output，空字典记录辅助输出，如各塔中间特征、范数、缩放 t、偏置 b；最终与两塔向量一起返回。它不是分类标签。
    out = {}
    # 【中文解释】out_dims 是维度配置的局部副本，复数 dims 提醒下面会统一成两个维度；self.out_dim 是类字段。
    out_dims = self.out_dim
    # 【中文解释】isinstance 是 Python 内置类型判断；若只给一个整数，下面复制为图像维度和文字维度的元组。
    if isinstance(out_dims, int):
      # 【中文解释】圆括号加逗号构造二元 tuple（元组）；索引 0 给图像塔，索引 1 给文本塔。例如 128 变成 (128,128)。
      out_dims = (out_dims, out_dims)

    # Embed the text:
    # 【中文解释】is not None 判断确实传了文字数据，不对数组做真假测试；JAX 数组不能当普通布尔值随意判断。只传图像时跳过文字塔。
    if text is not None:
      # 【中文解释】此处 text_model 局部变量将保存模型实例，不同于 self.text_model 路径字符串。import_module 导入模块，后面的 .Model 构建类实例。
      text_model = importlib.import_module(
          # 【中文解释】f-string 将配置路径插入模块名；例如得到 big_vision.models.proj.image_text.text_transformer。点号是包层级。
          f"big_vision.models.{self.text_model}"
      # 【中文解释】** 把字典展开成关键字参数；self.text or {} 在没有配置时用空字典；后展开的配置可覆盖 num_classes。这里 num_classes 实际控制输出投影维度，不意味着正在做类别预测。name="txt" 命名参数子树。
      ).Model(**{"num_classes": out_dims[1], **(self.text or {})}, name="txt")

      # 【中文解释】调用文字编码器，返回文字特征 ztxt [B,C] 和辅助字典 out_txt；逗号进行返回值解包。**kw 把额外参数继续传给子模型。
      ztxt, out_txt = text_model(text, **kw)
      # 【中文解释】dict.items() 迭代键值对；k=key（键），v=value（值），此处来自文字塔辅助输出，并非固定学习参数名。
      for k, v in out_txt.items():
        # 【中文解释】给辅助键添加 txt/ 前缀，避免与图像塔同名键冲突；字符串斜杠是命名习惯，不是在创建文件夹。
        out[f"txt/{k}"] = v

      # Normalize the embeddings the models give us.
      # 【中文解释】linalg=linear algebra（线性代数）；norm 默认 L2 范数。axis=1 沿 C 维计算每条文字向量长度，keepdims=True 保留单列，因此 [B,C]→[B,1]，便于后续广播除法。
      out["txt/norm"] = jnp.linalg.norm(ztxt, axis=1, keepdims=True)
      # 【中文解释】链式赋值把同一归一化结果存入 ztxt 和字典；[B,1] 范数广播到 [B,C]。1e-8 是很小的 ε，防零除。归一化后点积近似余弦相似度，而不是按词归一化。
      out["txt/normalized"] = ztxt = ztxt / (out["txt/norm"] + 1e-8)

    # 【中文解释】这里 image 是本次图片数据 [B,H,W,3]；如果只给文字就跳过图像塔，返回 zimg=None。
    if image is not None:
      # 【中文解释】动态选择图像编码模块，局部 image_model 随后保存模型实例；self.image_model 仍然是模块路径配置。
      image_model = importlib.import_module(
          # 【中文解释】例如拼出 big_vision.models.vit，导入视觉 Transformer 实现；这是从仓库选择代码，不是在线下载图片。
          f"big_vision.models.{self.image_model}"
      # 【中文解释】图像塔投影维度取 out_dims[0]；image 配置后展开，可覆盖默认项；name="img" 命名参数子树。pylint 注释只关闭静态风格警告，不参与模型计算。
      ).Model(**{"num_classes": out_dims[0], **(self.image or {})}, name="img")  # pylint: disable=not-a-mapping

      # 【中文解释】返回图像匹配向量 zimg [B,C] 和辅助字典 out_img；这里的全局匹配向量用于 SigLIP，不能当作 PaLI-3 输入 UL2 的全部未池化 patch tokens。
      zimg, out_img = image_model(image, **kw)
      # 【中文解释】重新提醒 k/v 是图像辅助输出的键和值；这次迭代来自 out_img，而不是上方的文字 out_txt。
      for k, v in out_img.items():
        # 【中文解释】img/ 前缀标记图像侧输出，与 txt/ 对称，方便训练器、评估器和日志区分。
        out[f"img/{k}"] = v

      # Normalize the embeddings the models give us.
      # 【中文解释】图像向量 [B,C] 沿特征维求 L2 长度，得到 [B,1]；jnp.linalg.norm 是 JAX 库函数。
      out["img/norm"] = jnp.linalg.norm(zimg, axis=1, keepdims=True)
      # 【中文解释】图像侧单位化，结果仍 [B,C]；广播规则与文字侧相同。1e-8 保证零向量附近除法有定义。
      out["img/normalized"] = zimg = zimg / (out["img/norm"] + 1e-8)

    # 【中文解释】temp=temperature，init=initial；取自然对数得到可学习 t 的初值，后面 exp(t) 保证实际缩放大于零。例如初始缩放 10，t 初值约 2.3026。
    temp_init = jnp.log(self.temperature_init)
    # 【中文解释】self.param 是 Flax 参数注册/读取方法：参数名 t 在 init 时创建，在 apply 时从变量树取得。t 是 log-scale 标量数组 [1]，不是文本 text。
    t = self.param("t",
                   # 【中文解释】lambda 是匿名初始化函数；key 是随机数键但这里不用随机性；shape 是尺寸，dtype 是数值类型。jnp.ones 生成全 1 数组，再乘初值。
                   lambda key, shape, dtype: temp_init * jnp.ones(shape, dtype),
                   # 【中文解释】形状参数 (1,) 表示单元素数组，float32 表示32位浮点；两处初始化都让缩放/偏置可以广播到匹配分数矩阵。
                   (1,), jnp.float32)
    # 【中文解释】exp 是 e 的幂，把无约束 log-scale t 变成正缩放因子；训练器取 extras["t"] 乘点积。out 是此前积累的辅助输出字典。
    out["t"] = jnp.exp(t)

    # 【中文解释】另存原始 log-scale 参数供日志读取；它与 out["t"] 的实际正缩放数值不同。
    out["t/parameter"] = t
    # 【中文解释】:= 是赋值表达式（海象运算符）：先把偏置初值赋给 b_init，再判断是不是 None；0.0 也会创建参数，因为判断不是 if b_init。
    if (b_init := self.bias_init) is not None:
      # 【中文解释】b=bias，共享偏置 [1]；lambda 的 k/s/d 是 key/shape/dtype 的短写，与上面的初始化参数作用相同。负偏置可调节大量负图文对的默认分数。
      out["b"] = self.param("b", lambda k, s, d: b_init * jnp.ones(s, d),
                            # 【中文解释】形状参数 (1,) 表示单元素数组，float32 表示32位浮点；两处初始化都让缩放/偏置可以广播到匹配分数矩阵。
                            (1,), jnp.float32)

    # We could actually play with pre-multiplying by temperature here, such
    # that out["t"] is nothing special to the trainer anymore.

    # 【中文解释】返回三元组：图像向量、文字向量、辅助字典；训练器将第三项命名 extras。缺少某侧输入时对应向量为 None；需要双塔损失时必须两侧都提供。
    return zimg, ztxt, out


# 【中文解释】load 是普通模块函数：init_params 为已初始化参数树，init_files 为检查点路径/字典，model_cfg 为模型配置；img/txt_load_kw 是图像/文字加载的额外关键字。默认 {} 是共享可变对象，此函数只展开它们，不应在扩展时随意原地修改。
def load(init_params, init_files, model_cfg, img_load_kw={}, txt_load_kw={}):  # pylint: disable=dangerous-default-value
  """Loads both towers, `init_files` is now a dict with `img` and `txt` keys."""
  # 【中文解释】检查点既允许字符串也允许字典；字符串可能是下方 VANITY_NAMES 中的快捷名称，或实际文件路径。
  if isinstance(init_files, str):
    # 【中文解释】dict.get(key,default) 查快捷名称映射，找不到则保留原字符串。VANITY_NAMES 表示便于记忆的别名，不能把名字中的规模视为当前已加载事实。
    init_files = VANITY_NAMES.get(init_files, init_files)

  # 【中文解释】检查点既允许字符串也允许字典；字符串可能是下方 VANITY_NAMES 中的快捷名称，或实际文件路径。
  if isinstance(init_files, str):
    # A shortcut for a single file checkpoint of a two_towers model.
    # 【中文解释】keys() 取得配置键；这里按键是否存在决定从联合检查点取 b，并非判断 bias_init 值是否为 None，使用配置时需要注意这一差别。
    if "bias_init" in model_cfg.keys():
      logging.info("loading img, txt, t, and b from a single checkpoint.")
      # 【中文解释】字典推导式给四个参数子树分别指定路径；path:subtree 是本仓库检查点读取约定，不是普通 Windows 文件名。
      init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t", "b")}
    else:
      logging.info("loading img, txt, and t from a single checkpoint.")
      # 【中文解释】无偏置分支只加载图像、文字和缩放 t；k 是这次推导的子树名称。
      init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t")}
  else:
    # 【中文解释】浅拷贝外层字典，后面 pop 删除键时不改调用者原字典；嵌套对象仍共享，浅拷贝不是 deepcopy。
    init_files = {**init_files}  # Shallow copy because we'll pop stuff off.

  # 【中文解释】空参数树/None 时补占位，用于交互环境；这不代表已经训练或已成功恢复全部参数。
  if not init_params:  # Convenience to skip checks in colab.
    # 【中文解释】只给两塔设占位 None，后续各塔 load 决定如何恢复；img/text 的值不是样本数据。
    init_params = {"img": None, "txt": None}
  # 【中文解释】restored=已恢复，params=parameters（参数）；先浅复制原始参数树，再用检查点覆盖需要的子树。
  restored_params = {**init_params}

  # 【中文解释】pop(key,default) 取值并删键；注意 Python 会先求 default 表达式，所以内层 pop("img",None) 即使存在 image 也会执行。两别名同时提供时都会被删，外层 image 值优先。
  img_init = init_files.pop("image", init_files.pop("img", None))
  # 【中文解释】图像检查点路径非空才恢复；若没有，则保留初始化图像参数。
  if img_init:
    restored_params["img"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('image_model', 'vit')}"
    ).load(init_params["img"], img_init, model_cfg.image, **img_load_kw)

  # 【中文解释】文字检查点也接受 text/txt 两种键；与图像一样，内层 pop 会提前执行，不是懒求值。
  txt_init = init_files.pop("text", init_files.pop("txt", None))
  # 【中文解释】存在文字检查点时调用文字模型自己的 load；不把图像权重装到文字塔里。
  if txt_init:
    restored_params["txt"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('text_model', 'proj.image_text.text_transformer')}"  # pylint: disable=line-too-long
    ).load(init_params["txt"], txt_init, model_cfg.text, **txt_load_kw)

  # 【中文解释】这里 t_init 是缩放参数检查点，不是初始数值 temperature_init；支持 temperature/t 两种路径键。
  t_init = init_files.pop("temperature", init_files.pop("t", None))
  # 【中文解释】有路径才覆盖 log-scale 参数；路径字符串的真假判断不等于参数值的真假判断。
  if t_init:
    # 【中文解释】utils.load_params 读取检查点中的参数数组；加载的是原始 t，模型前向仍需 exp(t)。
    restored_params["t"] = utils.load_params(t_init)

  # 【中文解释】这里 b_init 命名被复用于偏置检查点路径，与 __call__ 中的偏置数值初值不同；分属两个函数的局部作用域。
  b_init = init_files.pop("bias", init_files.pop("b", None))
  # 【中文解释】有偏置路径才加载；字符串为空时不执行恢复。
  if b_init:
    # 【中文解释】恢复共享偏置参数 b；参数树的键须与 self.param("b",...) 对应。
    restored_params["b"] = utils.load_params(b_init)

  # 【中文解释】assert 是断言：所有配置路径应被消费；剩余键提示可能拼错。以 python -O 运行可关闭断言，因此不能把它当可靠输入安全验证。
  assert not init_files, (
      f"There's something unused left in `config.model_init`. You probably got "
      f"a typo. Here it is: {init_files}")

  # 【中文解释】返回恢复后的参数树；并不返回模型实例，也不自动执行训练或推理。
  return restored_params


# Shortcut names for some canonical paper checkpoints:
# 【中文解释】快捷检查点名称→存储路径的常量字典；大写表示约定上不修改。下方 gs:// 是 Google Cloud Storage 对象路径；SigLIP2 条目不是本论文 SigLIP 的同一版本。
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # SigLIP image encoder checkpoints from https://arxiv.org/abs/2303.15343
    "SigLIP B/16 224": "gs://big_vision/siglip/webli_en_b16_224_63724782.npz",
    "SigLIP B/16 256": "gs://big_vision/siglip/webli_en_b16_256_60500360.npz",
    "SigLIP B/16 384": "gs://big_vision/siglip/webli_en_b16_384_68578854.npz",
    "SigLIP B/16 512": "gs://big_vision/siglip/webli_en_b16_512_68580893.npz",
    "SigLIP L/16 256": "gs://big_vision/siglip/webli_en_l16_256_60552751.npz",
    "SigLIP L/16 384": "gs://big_vision/siglip/webli_en_l16_384_63634585.npz",
    "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz",
    "SigLIP So400m/14 384": "gs://big_vision/siglip/webli_en_so400m_384_58765454.npz",
    "SigLIP B/16-i18n 256": "gs://big_vision/siglip/webli_i18n_b16_256_66117334.npz",

    # SigLIP 2 image and text encoder checkpoints from https://arxiv.org/abs/2502.14786
    "SigLIP2 B/16 224": "gs://big_vision/siglip2/siglip2_b16_224.npz",
    "SigLIP2 B/16 256": "gs://big_vision/siglip2/siglip2_b16_256.npz",
    "SigLIP2 B/16 384": "gs://big_vision/siglip2/siglip2_b16_384.npz",
    "SigLIP2 B/16 512": "gs://big_vision/siglip2/siglip2_b16_512.npz",
    "SigLIP2 B/32 256": "gs://big_vision/siglip2/siglip2_b32_256.npz",
    "SigLIP2 L/16 256": "gs://big_vision/siglip2/siglip2_l16_256.npz",
    "SigLIP2 L/16 384": "gs://big_vision/siglip2/siglip2_l16_384.npz",
    "SigLIP2 L/16 512": "gs://big_vision/siglip2/siglip2_l16_512.npz",
    "SigLIP2 So400m/14 224": "gs://big_vision/siglip2/siglip2_so400m14_224.npz",
    "SigLIP2 So400m/14 384": "gs://big_vision/siglip2/siglip2_so400m14_384.npz",
    "SigLIP2 So400m/16 256": "gs://big_vision/siglip2/siglip2_so400m16_256.npz",
    "SigLIP2 So400m/16 384": "gs://big_vision/siglip2/siglip2_so400m16_384.npz",
    "SigLIP2 So400m/16 512": "gs://big_vision/siglip2/siglip2_so400m16_512.npz",
    "SigLIP2 g-opt/16 256": "gs://big_vision/siglip2/siglip2_g-opt16_256.npz",
    "SigLIP2 g-opt/16 384": "gs://big_vision/siglip2/siglip2_g-opt16_384.npz",
    # SigLIP 2 NaFlex image and text encoder checkpoints.
    # These need `image_model="proj.image_text.naflex_vit"` for the image encoder
    # and a non-standard preprocessing, see configs/proj/image_text/README_siglip2.md.
    "SigLIP2 B/16 NaFlex": "gs://big_vision/siglip2/siglip2_b16_naflex.npz",
    "SigLIP2 So400m/16 NaFlex": "gs://big_vision/siglip2/siglip2_so400m16_naflex.npz",
    # pylint: enable=line-too-long
}
