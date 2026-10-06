# 当前文件让 clip/ 成为可导入的 Python 包。
# 前面的点表示“同一个包内的 clip.py”；* 把该模块允许公开的名字
# 导出到包顶层，所以外部可以写 import clip; clip.load(...)、clip.tokenize(...)。
# 这里的 * 不会执行一次 CLIP 训练，只会按 Python 导入规则载入模块。
from .clip import *
