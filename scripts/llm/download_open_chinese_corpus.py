from __future__ import annotations

"""
可选脚本：下载一份开源中文训练语料。

默认数据源：
    Hugging Face Datasets: wikimedia/wikipedia
    子集：20231101.zh

这份数据来自中文维基百科清洗文本。
它适合用来学习“从真实开源语料准备训练数据”的流程。

重要提醒：
1. 这个脚本默认只抽取前面一小部分文章，不会下载整个中文维基百科。
2. 训练出来的模型质量取决于语料数量和质量；少量文章只适合学习流程。
3. 使用开源语料时要尊重原始许可和署名要求。

运行示例：

    python scripts/llm/download_open_chinese_corpus.py --out data/text/raw/open_zh_wikipedia.txt --max-docs 200

然后你可以重新训练 tokenizer：

    python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --out artifacts/llm/tokenizer --vocab-size 8000
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Any


def clean_text(text: str) -> str:
    """做非常轻量的文本清洗。

    这里不做复杂清洗，只做三件事：
    1. 把不同系统的换行统一成 \\n。
    2. 去掉每一行前后的空白。
    3. 把连续很多空行压缩成最多一个空行。

    为什么不要清洗太狠？
    - 初学项目要先保持逻辑清楚。
    - 过度清洗可能误删有用文本。
    - 真正大规模语料清洗会是单独的大工程。
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_hf_dataset(dataset_name: str, subset: str, split: str, cache_dir: str | None):
    """延迟导入并加载 Hugging Face dataset。

    为什么在函数里 import datasets？
    - 如果用户还没安装依赖，我们可以给出更友好的错误提示。
    - 否则 Python 会在脚本刚启动时直接报 ModuleNotFoundError。
    """

    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        print(
            "缺少依赖 datasets。请先运行：\n\n"
            "    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise

    # streaming=True 表示流式读取。
    # 好处是不用一次性把整个数据集下载到内存或硬盘里。
    return load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )


def write_source_note(
    path: Path,
    dataset_name: str,
    subset: str,
    split: str,
    docs_written: int,
    chars_written: int,
) -> None:
    """写一个来源说明文件。

    训练文本本身保持干净，来源、许可、数量这些信息单独放在 .source.md。
    """

    note_path = path.with_suffix(".source.md")
    note_path.write_text(
        "\n".join(
            [
                "# 语料来源说明",
                "",
                f"- 输出文本：`{path.name}`",
                f"- 数据集：`{dataset_name}`",
                f"- 子集：`{subset}`",
                f"- split：`{split}`",
                f"- 写入文章数：`{docs_written}`",
                f"- 写入字符数：`{chars_written}`",
                "",
                "## 许可提醒",
                "",
                "该默认来源是 Hugging Face 上的 `wikimedia/wikipedia` 数据集。",
                "其数据卡说明原始文本来自 Wikipedia dumps，并说明原始文本使用 GFDL 和 Creative Commons Attribution-Share-Alike 3.0 License。",
                "如果你发布、分发或基于该语料训练模型，请自行确认并遵守相关许可、署名和 share-alike 要求。",
                "",
                "数据集页面：https://huggingface.co/datasets/wikimedia/wikipedia",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Download a small open Chinese corpus for LLM practice.")

    # 输出训练文本文件。
    parser.add_argument(
        "--out",
        default="data/text/raw/open_zh_wikipedia.txt",
        help="Output .txt file for downloaded corpus.",
    )

    # 默认使用 wikimedia/wikipedia。
    # 保留成参数，是为了以后你可以换其它 Hugging Face 数据集。
    parser.add_argument(
        "--dataset",
        default="wikimedia/wikipedia",
        help="Hugging Face dataset name.",
    )

    # 20231101.zh 是中文维基百科子集。
    parser.add_argument(
        "--subset",
        default="20231101.zh",
        help="Dataset subset/config name.",
    )

    # 大多数预训练语料只有 train split。
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split.",
    )

    # 数据集中正文所在字段。
    # wikimedia/wikipedia 使用 text 字段保存文章正文。
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field that contains article text.",
    )

    # 数据集中标题字段。
    # 如果没有标题字段，脚本也能运行，只是不写标题。
    parser.add_argument(
        "--title-field",
        default="title",
        help="Optional field that contains article title.",
    )

    # 最多写入多少篇文章。
    # 默认 200 篇，适合初学者先跑通流程。
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200,
        help="Maximum number of documents to write.",
    )

    # 太短的文章跳过。
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip documents shorter than this many characters after cleaning.",
    )

    # 可选：最多写入多少字符。
    # 0 表示不限制。
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Maximum total characters to write. 0 means no limit.",
    )

    # 可选：Hugging Face datasets 缓存目录。
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_hf_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    docs_written = 0
    chars_written = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for example in dataset:
            # example 是一篇文章，对 wikimedia/wikipedia 来说通常包含：
            # - id
            # - url
            # - title
            # - text
            example = dict(example)

            raw_text = example.get(args.text_field)
            if not isinstance(raw_text, str):
                continue

            text = clean_text(raw_text)
            if len(text) < args.min_chars:
                continue

            title_value: Any = example.get(args.title_field, "")
            title = clean_text(title_value) if isinstance(title_value, str) else ""

            # 每篇文章之间用一个空行分隔。
            # 加标题能让训练文本更自然，也方便你打开文件检查内容。
            if title:
                document = f"标题：{title}\n{text}"
            else:
                document = text

            next_chars = len(document)
            if args.max_chars > 0 and chars_written + next_chars > args.max_chars:
                break

            f.write(document)
            f.write("\n\n")

            docs_written += 1
            chars_written += next_chars

            if docs_written >= args.max_docs:
                break

    write_source_note(
        path=out_path,
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        docs_written=docs_written,
        chars_written=chars_written,
    )

    print(f"Corpus saved to: {out_path}")
    print(f"Source note saved to: {out_path.with_suffix('.source.md')}")
    print(f"Documents written: {docs_written}")
    print(f"Characters written: {chars_written}")

    if docs_written == 0:
        print(
            "没有写入任何文章。可以尝试降低 --min-chars，或检查数据集字段名。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
