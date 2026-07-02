from __future__ import annotations

"""
导入开源聊天 SFT 数据，并转换成标准 messages JSONL。

默认支持：

1. HuggingFaceTB/everyday-conversations-llama3.1-2k
   - Hugging Face 页面展示为多轮 role/content messages。
   - 主要是英文日常和基础科学对话，可用于扩充“正常沟通”能力。

注意：
- 开源数据的许可和用途需要你在下载前自行确认。
- 本脚本只做格式转换，不保证数据质量、事实正确性或安全边界。
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.sft_data import row_to_messages, validate_messages


SUPPORTED_DATASETS = {
    "everyday": {
        "path": "HuggingFaceTB/everyday-conversations-llama3.1-2k",
        "split": "train_sft",
    },
}


def normalize_messages(raw_messages: Any, system_prompt: str) -> list[dict[str, str]] | None:
    if not isinstance(raw_messages, list):
        return None

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in raw_messages:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            return None
        messages.append({"role": role, "content": content})

    if messages[-1]["role"] != "assistant":
        return None
    validate_messages(messages)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Import open SFT datasets into messages JSONL.")
    parser.add_argument("--source", choices=sorted(SUPPORTED_DATASETS), default="everyday")
    parser.add_argument("--out", default="data/text/raw/sft_open_everyday.jsonl")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant. Answer clearly, naturally, and briefly.",
    )
    args = parser.parse_args()

    meta = SUPPORTED_DATASETS[args.source]
    dataset = load_dataset(meta["path"], split=meta["split"], streaming=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in dataset:
            raw_messages = row.get("messages")
            messages = normalize_messages(raw_messages, args.system_prompt)
            if messages is None:
                continue

            candidate = {"messages": messages}
            # Reuse the same validator path as local SFT rows.
            if row_to_messages(candidate) is None:
                continue

            f.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if written >= args.max_rows:
                break

    print(f"Wrote {written} rows to {out_path}")
    if written == 0:
        raise SystemExit("No rows were imported. Check dataset schema or network access.")


if __name__ == "__main__":
    main()
