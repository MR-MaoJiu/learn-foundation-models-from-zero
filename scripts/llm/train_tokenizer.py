from __future__ import annotations

"""
第一步脚本：训练 tokenizer。

你可以把 tokenizer 理解成“文字和数字之间的翻译器”。

LLM 不能直接吃中文字符串，它只能吃数字 token id。
所以在训练模型之前，必须先有 tokenizer：

    中文文本 -> tokenizer -> token id 列表

例如：

    "语言模型" -> [120, 88, 341]

这个脚本会读取原始文本文件，学习一个 BPE tokenizer，
然后保存成：

    artifacts/llm/tokenizer/tokenizer.json

后面的数据处理、训练、生成都会用到同一个 tokenizer。
"""

import argparse
from pathlib import Path
import sys


# 当前文件位置：
#   learn-foundation-models-from-zero/scripts/llm/train_tokenizer.py
#
# Path(__file__).resolve() 得到当前脚本的绝对路径。
# parents[2] 表示往上三级，也就是项目根目录 learn-foundation-models-from-zero/。。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 把 learn-foundation-models-from-zero/src 加到 Python 模块搜索路径。
# 这样下面才能 import foundation_models.llm。
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.tokenizer import train_bpe_tokenizer


def main() -> None:
    """命令行入口。

    你在终端里运行：

        python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt --out artifacts/llm/tokenizer --vocab-size 2000

    argparse 会把这些命令行参数解析成 args。
    """

    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")

    # --input 可以接收一个或多个 txt 文件。
    #
    # nargs="+" 的意思是：至少 1 个，也可以多个。
    # 例如：
    #   --input a.txt b.txt c.txt
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more raw .txt files.",
    )

    # --out 是 tokenizer 输出目录。
    # 最终会保存为：<out>/tokenizer.json
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory, e.g. artifacts/llm/tokenizer.",
    )

    # --vocab-size 是词表大小。
    #
    # 词表越大：
    # - 单个 token 能表示更长的片段。
    # - 但模型输出层也会更大，参数更多。
    #
    # 小语料即使设置 8000，也可能训练不出完整 8000 个 token。
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
        help="Vocabulary size.",
    )

    args = parser.parse_args()

    # 真正的训练逻辑在 src/foundation_models/llm/tokenizer.py。
    # 这个脚本只负责读取命令行参数，然后调用函数。
    output = train_bpe_tokenizer(
        input_files=args.input,
        out_dir=args.out,
        vocab_size=args.vocab_size,
    )

    print(f"Tokenizer saved to: {output}")


# 只有直接运行这个文件时，才执行 main()。
#
# 如果别的 Python 文件 import 这个脚本，
# 下面这段不会自动执行。
if __name__ == "__main__":
    main()
