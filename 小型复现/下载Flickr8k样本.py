"""从 Hugging Face 的 Flickr8k 镜像取少量真实图片和说明文字。"""

# argparse 让用户通过 --train、--validation 控制下载数量。
import argparse
# BytesIO 把下载的图片字节当作文件交给 Pillow。
from io import BytesIO
# json 负责把图片文件名和描述文字写入清单。
import json
# pathlib 负责处理 Windows 和其他系统上的文件路径。
from pathlib import Path
# shutil 查找系统中是否安装 curl 命令。
import shutil
# subprocess 以参数列表调用 curl，URL 中的特殊字符不会被当成命令执行。
import subprocess
# time 在临时网络错误后稍等再试。
import time
# urlencode 把数据集参数安全地拼入 URL。
from urllib.parse import urlencode

# Pillow 读取图片并缩放到教学用尺寸。
from PIL import Image

# Hugging Face 的官方 Dataset Viewer 每次最多返回 100 条。
API_URL = "https://datasets-server.huggingface.co/rows"


def fetch(url: str, params: dict | None = None) -> bytes:
    """使用系统 curl 请求数据或图片，返回原始字节。"""
    # 有查询参数时，先把它们变成 URL 后面的 ?key=value 部分。
    if params is not None:
        # urlencode 会正确处理斜杠、空格等特殊字符。
        url = url + "?" + urlencode(params)
    # Windows 自带 curl.exe；其他系统通常叫 curl。
    curl = shutil.which("curl.exe") or shutil.which("curl")
    # 没找到 curl 时给出明确提示。
    if curl is None:
        # 不要继续运行，否则错误会难以理解。
        raise RuntimeError("未找到 curl；请先安装 curl")
    # -f 让 HTTP 错误返回失败，-L 跟随跳转，-sS 只在出错时显示说明。
    command = [curl, "-fLsS", "--retry", "2", "--max-time", "30", url]
    # TLS 握手偶尔中断时，重新启动 curl 再试一次。
    for attempt in range(2):
        # capture_output 把响应内容留在内存，不把二进制图片打印出来。
        result = subprocess.run(command, capture_output=True)
        # 退出码为 0 表示成功，stdout 就是 JSON 或图片字节。
        if result.returncode == 0:
            # 把数据交给调用者。
            return result.stdout
        # 最后一次失败时只报告退出码，避免把很长的签名 URL 打印出来。
        if attempt == 1:
            # 给初学者一个明确的网络错误说明。
            raise RuntimeError(f"网络下载失败，curl 退出码 {result.returncode}；请稍后重新运行")
        # 每次多等一秒再试。
        time.sleep(attempt + 1)
    # 实际执行不会到这里，只为让返回路径完整。
    raise RuntimeError("网络下载失败")


def save_split(split_name: str, limit: int, output_root: Path) -> None:
    """按页读取一个划分，保存前 limit 张图片和它们的五条描述。"""
    # 训练集和验证集使用各自的图片子目录。
    image_dir = output_root / split_name
    # 如果目录不存在就创建；已存在时继续使用。
    image_dir.mkdir(parents=True, exist_ok=True)
    # 清单记录每张图的相对路径和五条文字描述。
    manifest = []
    # 多读一些候选行，以便个别图片下载失败时仍可凑够目标数量。
    max_candidates = min(6000 if split_name == "train" else 1000, limit * 2)
    # API 每页至多 100 条，按 0、100、200 等起点循环取页。
    for offset in range(0, max_candidates, 100):
        # 已经保存到指定数量就结束本划分。
        if len(manifest) >= limit:
            break
        # 末页可能不足 100 条，所以只请求剩余候选数。
        page_length = min(100, max_candidates - offset)
        # 这些参数告诉 API 我们要哪个数据集、划分和行范围。
        params = {"dataset": "jxie/flickr8k", "config": "default", "split": split_name, "offset": offset, "length": page_length}
        # 把 API 的 JSON 响应转换成 Python 字典。
        page = json.loads(fetch(API_URL, params=params))
        # 遍历这一页的每条图文数据。
        for item in page["rows"]:
            # 目标数量够了就不再继续下载本页的图片。
            if len(manifest) >= limit:
                break
            # row_idx 是数据集中的原始行号，可用于生成稳定文件名。
            index = item["row_idx"]
            # row 含图片链接和五个 caption 字段。
            row = item["row"]
            # 固定图片名便于按编号找到它。
            image_name = f"{index:05d}.jpg"
            # 完整图片若已存在，重新运行时可以沿用它。
            image_path = image_dir / image_name
            # 个别远端图片可能暂时不可访问，跳过后继续取下一行。
            try:
                # 只在图片不存在或文件为空时发起图片请求。
                if not image_path.exists() or image_path.stat().st_size == 0:
                    # API 返回短期有效的图片链接，此处马上读取图片字节。
                    image_bytes = fetch(row["image"]["src"])
                    # 用 Pillow 从内存中的图片字节解码，并转成 RGB。
                    with Image.open(BytesIO(image_bytes)) as source:
                        # resize 只用于教学样本；缩小图片可以降低磁盘占用。
                        image = source.convert("RGB").resize((224, 224))
                    # 先写到临时文件，避免下载中断留下一个看似完整的 JPEG。
                    temporary_path = image_path.with_suffix(".tmp")
                    # Pillow 根据显式 format 参数编码成 JPEG。
                    image.save(temporary_path, format="JPEG", quality=90)
                    # 写完后再把临时文件改成正式图片名。
                    temporary_path.replace(image_path)
            except RuntimeError:
                # 显示出错图片的行号，方便将来重试。
                print(f"{split_name}: 跳过暂时无法下载的第 {index} 行")
                # 不把缺失图片写入 JSON 清单。
                continue
            # Flickr8k 镜像中每张图片有五条文字描述。
            captions = [row[f"caption_{number}"] for number in range(5)]
            # 清单中的路径相对于 output_root，搬动整个文件夹也仍然有效。
            manifest.append({"image": f"{split_name}/{image_name}", "captions": captions})
    # 每个划分分别保存一个 JSON 文件，中文不会被转成转义符。
    with (output_root / f"{split_name}.json").open("w", encoding="utf-8") as handle:
        # indent=2 让初学者直接打开文件时更容易阅读。
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    # 打印实际保存的数量，方便核对是否取样成功。
    print(f"{split_name}: 保存 {len(manifest)} 张图片到 {image_dir}")
    # 如果实际数量不足，给出明确说明，避免误以为拿到了全部数据。
    if len(manifest) < limit:
        # 用户可以稍后再次运行，程序会沿用已经下载的文件。
        print(f"{split_name}: 目标 {limit} 张，当前还差 {limit - len(manifest)} 张；可稍后重试")


if __name__ == "__main__":
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="下载少量 Flickr8k 图文样本")
    # 默认训练样本 64 张，仍保持适合入门的小规模。
    parser.add_argument("--train", type=int, default=64)
    # 默认验证样本 16 张，与训练集分开。
    parser.add_argument("--validation", type=int, default=16)
    # 读取用户给出的两个数量。
    args = parser.parse_args()
    # 防止输入 0 或负数导致空数据集。
    if args.train < 1 or args.validation < 1:
        # 明确报错，方便初学者知道参数哪里不对。
        parser.error("--train 和 --validation 都必须大于 0")
    # 数据默认放在当前脚本旁边的 real_data/flickr8k-mini 文件夹。
    output_root = Path(__file__).resolve().parent / "real_data" / "flickr8k-mini"
    # 创建数据集根目录。
    output_root.mkdir(parents=True, exist_ok=True)
    # 先保存训练样本。
    save_split("train", args.train, output_root)
    # 再保存独立的验证样本。
    save_split("validation", args.validation, output_root)
