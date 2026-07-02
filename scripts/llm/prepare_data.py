from __future__ import annotations

"""
第二步脚本：把原始文本变成训练数据。

训练 LLM 需要的不是普通 txt，而是一长串 token id。

这个脚本做的事情：

1. 读取原始中文文本。
2. 用 tokenizer 把文本转换成 token id。
3. 切分训练集和验证集。
4. 保存成二进制文件：
       data/text/processed/train.bin
       data/text/processed/val.bin

为什么要有 train.bin 和 val.bin？

- train.bin：用于训练，模型会根据它更新参数。
- val.bin：用于验证，只看效果，不更新参数。

验证集可以帮助你判断模型是不是只是在死记硬背训练文本。
"""

import argparse
from pathlib import Path
import sys


# 找到项目根目录，并把 src 加入导入路径。
# 这样脚本可以 import src/foundation_models/llm 里的代码。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.data import make_train_val_bins


def main() -> None:
    """命令行入口。

    示例命令：

        python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
    """

    parser = argparse.ArgumentParser(description="Convert raw text into train.bin and val.bin.")

    # 一个或多个原始 txt 文件。
    # 你以后可以把更多语料文件放进 data/text/raw/，然后一起传进来。
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more raw .txt files.",
    )

    # tokenizer.json 的路径。
    # 必须和训练模型、生成文本时使用的是同一个 tokenizer。
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer.json.",
    )

    # 输出目录。
    # 脚本会在这里生成 train.bin 和 val.bin。
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for .bin files.",
    )

    # 验证集比例。
    # 0.1 表示拿 10% token 做验证集，90% 做训练集。
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction used as validation data.",
    )

    # 随机种子。
    # 用来控制打乱语料的随机性，方便你复现实验。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )

    args = parser.parse_args()

    # 真正的数据处理逻辑在 src/foundation_models/llm/data.py。
    # 这里把命令行参数传进去。
    train_path, val_path = make_train_val_bins(
        input_files=args.input,
        tokenizer_path=args.tokenizer,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Train tokens saved to: {train_path}")
    print(f"Val tokens saved to:   {val_path}")


if __name__ == "__main__":
    main()
