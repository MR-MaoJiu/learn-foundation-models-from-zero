from __future__ import annotations

"""
从 MNIST 图像分类数据构造图文配对数据。

真实的多模态数据通常是“图片 + 标题”或“图片 + 描述”。
为了学习原理，我们先用 MNIST 做一个可控的小例子：

图片：一张手写数字 7
文本：这是一张手写数字七。

这样你不用一开始处理复杂的网页图片和噪声文本，也能看懂图文对齐是怎么训练的。
"""

import argparse
import csv
import json
from pathlib import Path


ZH_DIGITS = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}


def make_text(label: int) -> str:
    """根据数字标签生成一条中文描述。"""

    return f"这是一张手写数字{ZH_DIGITS[label]}，也就是阿拉伯数字{label}。"


def convert_split(csv_path: Path, out_path: Path) -> int:
    """把一个 MNIST labels.csv 转成图文对 JSONL。

    参数：
        csv_path: 视觉数据里的 labels.csv。
        out_path: 要写出的 JSONL 文件路径。

    JSONL 每一行都是一个 JSON 对象，适合训练时逐行读取。
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}。请先运行：python scripts/vision/download_open_vision_dataset.py"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        root = csv_path.parent

        for row in reader:
            label = int(row["label"])

            # row["image"] 是相对 labels.csv 所在目录的路径，例如 images/000001.png。
            image_path = root / row["image"]
            text = make_text(label)

            item = {
                "image": str(image_path),
                "text": text,
                "label": label,
            }
            f_out.write(json.dumps(item, ensure_ascii=False))
            f_out.write("\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MNIST image-text pairs for multimodal learning.")
    parser.add_argument("--mnist-dir", default="data/vision/mnist", help="Directory created by the vision downloader.")
    parser.add_argument("--out", default="data/multimodal/mnist_pairs", help="Output directory for JSONL files.")
    args = parser.parse_args()

    mnist_dir = Path(args.mnist_dir)
    out_dir = Path(args.out)

    train_count = convert_split(mnist_dir / "train" / "labels.csv", out_dir / "train.jsonl")
    val_count = convert_split(mnist_dir / "val" / "labels.csv", out_dir / "val.jsonl")

    (out_dir / "SOURCE.md").write_text(
        "\n".join(
            [
                "# MNIST 图文配对来源说明",
                "",
                "图像来自 `ylecun/mnist`；文本描述由本项目根据数字标签自动生成。",
                "该数据用于学习图文对齐，不是真实自然图片字幕数据集。",
                "",
                f"- 训练配对数：`{train_count}`",
                f"- 验证配对数：`{val_count}`",
                "- MNIST 页面：https://huggingface.co/datasets/ylecun/mnist",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Image-text pairs saved to: {out_dir}")
    print(f"Train pairs: {train_count}")
    print(f"Val pairs:   {val_count}")


if __name__ == "__main__":
    main()
