from __future__ import annotations

"""
下载并整理开源视觉训练素材：MNIST。

MNIST 是最经典的手写数字图片数据集之一：
- 图片内容：0 到 9 的手写数字。
- 图片大小：28 x 28。
- 图片类型：灰度图，也就是只有一个颜色通道。
- 学习任务：给模型一张图片，让模型判断它是哪一个数字。

为什么这个脚本要把数据导出成 PNG 和 labels.csv？

很多公开数据集会以专门的机器学习格式保存，适合程序读取，但不适合初学者观察。
这里我们故意保存成普通图片和 CSV 文件，这样你可以直接打开目录，看见训练素材长什么样。
"""

import argparse
import csv
from pathlib import Path
import sys


def load_mnist(split: str):
    """从 Hugging Face Datasets 加载 MNIST 的一个 split。

    参数：
        split: 数据集划分名称。MNIST 常用 `train` 和 `test`。

    返回：
        一个 Hugging Face Dataset 对象。你可以把它理解成“很多条样本组成的列表”。

    为什么在函数里面 import datasets？
        这样即使用户还没安装 datasets，也能给出更友好的报错提示。
    """

    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        print("缺少 datasets 依赖。请先运行：pip install -r requirements.txt", file=sys.stderr)
        raise

    return load_dataset("ylecun/mnist", split=split)


def export_split(dataset, out_dir: Path, max_samples: int) -> int:
    """把一个数据集划分导出成 PNG 图片和 labels.csv。

    参数：
        dataset: Hugging Face Dataset，例如 MNIST 的 train split。
        out_dir: 输出目录，例如 data/vision/mnist/train。
        max_samples: 最多导出多少张图片。学习阶段不需要一开始导出全部数据。

    返回：
        实际导出的样本数量。

    输出结构：
        out_dir/
          labels.csv
          images/
            000000.png
            000001.png
    """

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "labels.csv"
    count = 0

    # newline="" 是 csv 模块推荐写法，可以避免不同系统上出现多余空行。
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"])
        writer.writeheader()

        for example in dataset:
            if count >= max_samples:
                break

            # example["image"] 是 PIL Image 对象。
            # convert("L") 表示转成灰度图，L 可以理解为 luminance。
            image = example["image"].convert("L")
            label = int(example["label"])

            # 用固定 6 位编号命名，文件排序时更整齐。
            file_name = f"{count:06d}.png"
            image.save(images_dir / file_name)

            # CSV 里保存相对路径，方便整个目录移动。
            writer.writerow({"image": f"images/{file_name}", "label": label})
            count += 1

    return count


def write_source_note(out_dir: Path, train_count: int, val_count: int) -> None:
    """写一份数据来源说明，方便以后回看数据从哪里来。"""

    (out_dir / "SOURCE.md").write_text(
        "\n".join(
            [
                "# MNIST 来源说明",
                "",
                "- 数据集：`ylecun/mnist`",
                "- 页面：https://huggingface.co/datasets/ylecun/mnist",
                "- 任务：手写数字图像分类",
                f"- 导出训练样本数：`{train_count}`",
                f"- 导出验证样本数：`{val_count}`",
                "",
                "Hugging Face 数据集卡说明 MNIST 包含 70,000 张 28x28 黑白手写数字图像，",
                "其中训练集 60,000 张，测试集 10,000 张；许可信息标为 MIT Licence。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """命令行入口。

    argparse 会把用户在命令行里传入的参数解析成 args。
    如果你不传参数，就会使用 default 里的默认值。
    """

    parser = argparse.ArgumentParser(description="Download MNIST as open vision training material.")
    parser.add_argument("--out", default="data/vision/mnist", help="Output directory.")
    parser.add_argument("--max-train", type=int, default=2000, help="Maximum train images to export.")
    parser.add_argument("--max-val", type=int, default=500, help="Maximum validation images to export.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # MNIST 原始划分叫 train/test。
    # 为了和训练代码统一，我们把 test 导出成 val，表示 validation。
    train = load_mnist("train")
    val = load_mnist("test")

    train_count = export_split(train, out_dir / "train", args.max_train)
    val_count = export_split(val, out_dir / "val", args.max_val)
    write_source_note(out_dir, train_count, val_count)

    print(f"MNIST exported to: {out_dir}")
    print(f"Train images: {train_count}")
    print(f"Val images:   {val_count}")


if __name__ == "__main__":
    main()
