from __future__ import annotations

"""
端到端聊天模型工作流。

这个脚本模拟真实项目里的 pipeline：

1. 准备 tokenizer
2. 准备预训练 token 数据
3. 预训练 base model
4. SFT 对齐成 chat model
5. 冒烟评估
6. 导出部署包
"""

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def existing_inputs() -> list[str]:
    """返回 tokenizer 可读取的文本资产。

    tokenizer 可以看普通预训练文本，也可以看 SFT JSONL，让中文、技术词和
    ChatML 特殊 token 都进入同一套词表。真正的预训练 bin 只使用普通 `.txt`。
    """

    candidates = [
        "data/text/raw/tiny_zh_corpus.txt",
        "data/text/raw/open_zh_wikipedia.txt",
        "data/text/raw/sft_chat_zh.jsonl",
    ]
    return [path for path in candidates if (PROJECT_ROOT / path).exists()]


def existing_pretrain_texts() -> list[str]:
    """返回预训练使用的普通文本语料。

    SFT JSONL 不放进预训练 bin。聊天能力由 SFT 学，预训练只学通用文本分布。
    """

    candidates = [
        "data/text/raw/tiny_zh_corpus.txt",
        "data/text/raw/open_zh_wikipedia.txt",
    ]
    return [path for path in candidates if (PROJECT_ROOT / path).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full LLM training-to-deployment workflow.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--pretrain-config", default="configs/llm/debug.json")
    parser.add_argument("--sft-config", default="configs/llm/sft_debug.json")
    parser.add_argument("--export-dir", default="deployments/llm/chat_debug")
    parser.add_argument("--skip-pretrain", action="store_true", help="Reuse an existing base checkpoint.")
    parser.add_argument("--skip-sft", action="store_true", help="Reuse an existing SFT checkpoint.")
    args = parser.parse_args()

    inputs = existing_inputs()
    if not inputs:
        raise SystemExit("No text inputs found under data/text/raw.")

    run_command(
        [
            args.python,
            "scripts/llm/validate_sft_data.py",
            "--input",
            "data/text/raw/sft_chat_zh.jsonl",
            "--show-template",
        ]
    )

    run_command(
        [
            args.python,
            "scripts/llm/train_tokenizer.py",
            "--input",
            *inputs,
            "--out",
            "artifacts/llm/tokenizer",
            "--vocab-size",
            "1024",
            "--min-frequency",
            "1",
        ]
    )

    pretrain_inputs = existing_pretrain_texts()
    if not pretrain_inputs:
        raise SystemExit("No pretraining .txt inputs found under data/text/raw.")

    run_command(
        [
            args.python,
            "scripts/llm/prepare_data.py",
            "--input",
            *pretrain_inputs,
            "--tokenizer",
            "artifacts/llm/tokenizer/tokenizer.json",
            "--out",
            "data/text/processed",
            "--val-ratio",
            "0.1",
        ]
    )

    if not args.skip_pretrain:
        run_command([args.python, "scripts/llm/train.py", "--config", args.pretrain_config])

    if not args.skip_sft:
        run_command([args.python, "scripts/llm/train_sft.py", "--config", args.sft_config])

    sft_checkpoint = "checkpoints/llm/sft_debug/last.pt"
    run_command([args.python, "scripts/llm/evaluate_chat.py", "--checkpoint", sft_checkpoint])
    run_command(
        [
            args.python,
            "scripts/llm/export_model.py",
            "--checkpoint",
            sft_checkpoint,
            "--out",
            args.export_dir,
        ]
    )

    print("\nWorkflow done.")
    print(f"Serve with: {args.python} scripts/llm/serve_chat.py --model-dir {args.export_dir}")


if __name__ == "__main__":
    main()
