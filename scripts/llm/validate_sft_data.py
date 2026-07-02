from __future__ import annotations

"""
验证 SFT JSONL 语料格式。

这个脚本不训练模型，只检查数据是否适合“正常聊天模型”的 SFT：

1. 每行必须是 JSON。
2. 推荐使用 OpenAI/GPT 风格 `messages`。
3. system 只能放第一条。
4. user 和 assistant 必须交替。
5. 最后一条必须是 assistant，因为它是当前样本的训练目标。
6. 内容不能为空，且不能明显像乱码。
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.sft_data import row_to_sft_example, row_to_messages, validate_messages


MOJIBAKE_MARKERS = ("Ã", "Â", "å", "ä", "æ", "ç", "ï¼", "ã€")


def looks_like_mojibake(text: str) -> bool:
    """粗略识别 UTF-8 被错误当成其他编码读写后的乱码。"""

    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return marker_count >= 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate chat SFT JSONL data.")
    parser.add_argument("--input", required=True, help="Path to SFT JSONL.")
    parser.add_argument("--show-template", action="store_true", help="Print the first rendered ChatML sample.")
    args = parser.parse_args()

    input_path = Path(args.input)
    errors: list[str] = []
    warnings: list[str] = []
    total = 0
    rendered_preview = ""

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{line_number}: invalid JSON: {exc}")
                continue

            messages = row_to_messages(row)
            if messages is None:
                errors.append(f"{line_number}: row cannot be converted to messages")
                continue

            try:
                validate_messages(messages)
            except ValueError as exc:
                errors.append(f"{line_number}: {exc}")
                continue

            for index, message in enumerate(messages):
                content = message["content"]
                if looks_like_mojibake(content):
                    warnings.append(f"{line_number}: message {index} may contain mojibake")

            if args.show_template and not rendered_preview:
                example = row_to_sft_example(row)
                if example is not None:
                    rendered_preview = example.prompt + "\n" + example.response

    print(f"Checked {total} SFT rows: {input_path}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more")

    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
        raise SystemExit(1)

    if rendered_preview:
        print("\nRendered ChatML preview:")
        print(rendered_preview)

    print("SFT data format looks valid.")


if __name__ == "__main__":
    main()
