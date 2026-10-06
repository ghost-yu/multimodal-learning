"""两个教学模型共用的四类图文数据；每一步都写明为什么这样做。"""

# 导入 PyTorch，它负责张量运算和后续模型训练。
import torch

# 图片固定为 16×16，方便把图片分成 4×4 的小块。
IMAGE_SIZE = 16
# 四个类别依次是红方、红圆、蓝方、蓝圆。
CLASS_NAMES = ("红色方形", "红色圆形", "蓝色方形", "蓝色圆形")
# 数字是文字 token 的编号；0 留给填充符，虽然本项目不会用到填充。
VOCAB = {"PAD": 0, "CLS": 1, "SEP": 2, "RED": 3, "BLUE": 4, "SQUARE": 5, "CIRCLE": 6}


def make_images(class_ids: torch.Tensor) -> torch.Tensor:
    """根据类别编号画图片；输入 [B]，输出 [B,3,16,16]。"""
    # B 是当前这批图片的张数。
    batch_size = class_ids.shape[0]
    # 先创建全黑图片；3 代表红、绿、蓝三个颜色通道。
    images = torch.zeros(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    # 建立所有像素的行坐标和列坐标，二者形状都是 [16,16]。
    yy, xx = torch.meshgrid(torch.arange(IMAGE_SIZE), torch.arange(IMAGE_SIZE), indexing="ij")
    # 遍历这一批中每张图片的类别编号。
    for index, class_id in enumerate(class_ids.tolist()):
        # 整数除以 2：前两类是红色，后两类是蓝色。
        channel = 0 if class_id // 2 == 0 else 2
        # 对 2 取余：偶数类是方形，奇数类是圆形。
        is_circle = class_id % 2 == 1
        # 方形要求横纵坐标都在中心区域。
        square_mask = (xx >= 4) & (xx < 12) & (yy >= 4) & (yy < 12)
        # 圆形要求像素到中心的距离平方小于半径平方。
        circle_mask = (xx - 7.5).square() + (yy - 7.5).square() <= 16
        # 根据类别选方形或圆形像素掩码。
        mask = circle_mask if is_circle else square_mask
        # 把对应颜色通道的目标形状涂成 1.0。
        images[index, channel, mask] = 1.0
    # 给图片加一点随机噪声，避免训练只记住完全一样的像素。
    images += torch.rand_like(images) * 0.04
    # 保证所有像素值仍处于 [0,1]。
    return images.clamp(0.0, 1.0)


def make_text(class_ids: torch.Tensor) -> torch.Tensor:
    """把类别编号转换成四个文字 token；输入 [B]，输出 [B,4]。"""
    # 编号 0、1 对应红色，编号 2、3 对应蓝色。
    color_ids = torch.where(class_ids < 2, VOCAB["RED"], VOCAB["BLUE"])
    # 偶数编号对应方形，奇数编号对应圆形。
    shape_ids = torch.where(class_ids % 2 == 0, VOCAB["SQUARE"], VOCAB["CIRCLE"])
    # 每句话前面都放一个 CLS token，后面都放一个 SEP token。
    cls_ids = torch.full_like(class_ids, VOCAB["CLS"])
    # SEP 是句子结束标记。
    sep_ids = torch.full_like(class_ids, VOCAB["SEP"])
    # dim=1 表示沿“句子长度”方向拼接四列整数。
    return torch.stack((cls_ids, color_ids, shape_ids, sep_ids), dim=1)


def make_pairs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造四个正例和四个负例，返回图片、文字、二分类标签。"""
    # 四个类别各取一次，保证对比学习的一批里没有重复类别。
    class_ids = torch.randperm(4)
    # 根据四个类别分别生成图片和正确文字。
    positive_images = make_images(class_ids)
    positive_texts = make_text(class_ids)
    # 把类别整体错开一位；每条错配文字都与原图片不同类。
    negative_texts = make_text((class_ids + 1) % 4)
    # 同一张图片既与正确文字配对，也与错误文字配对。
    images = torch.cat((positive_images, positive_images), dim=0)
    # 前四条是正确文字，后四条是错误文字。
    texts = torch.cat((positive_texts, negative_texts), dim=0)
    # 1 表示匹配，0 表示不匹配。
    labels = torch.cat((torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)))
    # 模型若在 GPU 上，就把三个输入张量一起移到 GPU。
    return images.to(device), texts.to(device), labels.to(device)
