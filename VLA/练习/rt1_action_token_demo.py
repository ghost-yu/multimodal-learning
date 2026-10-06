"""RT-1 动作离散化的入门演示；仅用 Python 标准库。

这不是完整的 RT-1：它只展示一个连续动作数值如何对应到 256 个区间。
"""

NUM_BINS = 256


def encode(value: float, minimum: float, maximum: float) -> int:
    """把连续动作映射到 0..255 中的一个类别。"""
    if maximum <= minimum:
        raise ValueError("maximum 必须大于 minimum")
    if not minimum <= value <= maximum:
        raise ValueError("动作超出了允许范围")
    index = int((value - minimum) / (maximum - minimum) * NUM_BINS)
    return min(index, NUM_BINS - 1)


def decode(index: int, minimum: float, maximum: float) -> float:
    """用所在区间的中心，近似还原连续动作。"""
    if not 0 <= index < NUM_BINS:
        raise ValueError("类别编号必须在 0..255 之间")
    width = (maximum - minimum) / NUM_BINS
    return minimum + (index + 0.5) * width


if __name__ == "__main__":
    minimum, maximum = -1.0, 1.0
    for action in (-1.0, -0.25, 0.0, 0.34, 1.0):
        token = encode(action, minimum, maximum)
        recovered = decode(token, minimum, maximum)
        print(f"action {action:>5.2f} -> token {token:>3} -> decoded {recovered:>8.5f}")
