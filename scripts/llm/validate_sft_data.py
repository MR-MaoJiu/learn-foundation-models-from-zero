from __future__ import annotations

"""
Validate chat SFT JSONL data.

This script does not train a model. It checks whether the corpus is suitable
for normal chat-model SFT:
1. Each non-empty line must be valid JSON.
2. Each row must use OpenAI/GPT-style `messages`.
3. `system` can appear only as the first message, at most once.
4. `user` and `assistant` must alternate.
5. The final message must be `assistant`, because it is the supervised target.
6. Content must be non-empty and should not look like obvious mojibake.
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.sft_data import row_to_sft_example, row_to_messages, validate_messages


MOJIBAKE_MARKERS = (
    "锟",
    "�",
    "Ã",
    "Â",
    "å",
    "æ",
    "ç",
    "ä",
    "鈥",
    "銆",
    "绔",
    "鍙",
    "璇",
)


def looks_like_mojibake(text: str) -> bool:
    """Heuristically detect common UTF-8 decoding damage."""

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
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{line_number}: invalid JSON: {exc}")
                continue

            messages = row_to_messages(item)
            if messages is None:
                errors.append(f"{line_number}: row cannot be converted to messages")
                continue

            try:
                validate_messages(messages)
            except ValueError as exc:
                errors.append(f"{line_number}: {exc}")
                continue

            for index, message in enumerate(messages):
                if looks_like_mojibake(message["content"]):
                    warnings.append(f"{line_number}: message {index} may contain mojibake")

            if args.show_template and not rendered_preview:
                example = row_to_sft_example(item)
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
