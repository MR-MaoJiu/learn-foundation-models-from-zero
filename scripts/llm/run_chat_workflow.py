from __future__ import annotations

"""
End-to-end chat model workflow.

This script simulates a compact real-world LLM pipeline:
1. Build structured SFT chat data.
2. Validate the SFT JSONL format.
3. Build a larger plain-text pretraining corpus.
4. Train a tokenizer.
5. Convert pretraining text into train/val token binaries.
6. Pretrain a base model.
7. Run SFT to obtain a chat model.
8. Run a smoke evaluation.
9. Export a deployment package.

The default corpus is intentionally synthetic and small enough for learning,
but it is no longer the tiny seed file that caused validation data to be too
short for `max_seq_len=1024`.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def checkpoint_from_config(config_path: str) -> str:
    """Return `<training.output_dir>/last.pt` from a training config."""

    with (PROJECT_ROOT / config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    output_dir = config["training"]["output_dir"]
    return str(Path(output_dir) / "last.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full LLM training-to-deployment workflow.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--pretrain-config", default="configs/llm/debug.json")
    parser.add_argument("--sft-config", default="configs/llm/sft_debug.json")
    parser.add_argument("--export-dir", default="deployments/llm/chat_debug")
    parser.add_argument("--sft-count", type=int, default=10000, help="Number of synthetic SFT rows to generate.")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Tokenizer vocabulary size.")
    parser.add_argument(
        "--pretrain-count",
        type=int,
        default=8000,
        help="Number of synthetic plain-text pretraining paragraphs to add.",
    )
    parser.add_argument("--skip-pretrain", action="store_true", help="Reuse an existing base checkpoint.")
    parser.add_argument("--skip-sft", action="store_true", help="Reuse an existing SFT checkpoint.")
    args = parser.parse_args()

    needs_sft_data = not args.skip_sft or not args.skip_pretrain
    if needs_sft_data:
        run_command(
            [
                args.python,
                "scripts/llm/build_sft_chat_corpus.py",
                "--out",
                "data/text/raw/sft_chat_zh.jsonl",
                "--count",
                str(args.sft_count),
                "--seed",
                "42",
            ]
        )

        run_command(
            [
                args.python,
                "scripts/llm/validate_sft_data.py",
                "--input",
                "data/text/raw/sft_chat_zh.jsonl",
                "--show-template",
            ]
        )

    if not args.skip_pretrain:
        run_command(
            [
                args.python,
                "scripts/llm/build_pretrain_corpus.py",
                "--out",
                "data/text/raw/pretrain_zh.txt",
                "--seed-text",
                "data/text/raw/tiny_zh_corpus.txt",
                "--sft-jsonl",
                "data/text/raw/sft_chat_zh.jsonl",
                "--count",
                str(args.pretrain_count),
                "--max-sft-rows",
                str(args.sft_count),
                "--seed",
                "42",
            ]
        )

        run_command(
            [
                args.python,
                "scripts/llm/train_tokenizer.py",
                "--input",
                "data/text/raw/pretrain_zh.txt",
                "data/text/raw/sft_chat_zh.jsonl",
                "--out",
                "artifacts/llm/tokenizer",
                "--vocab-size",
                str(args.vocab_size),
                "--min-frequency",
                "1",
            ]
        )

        run_command(
            [
                args.python,
                "scripts/llm/prepare_data.py",
                "--input",
                "data/text/raw/pretrain_zh.txt",
                "--tokenizer",
                "artifacts/llm/tokenizer/tokenizer.json",
                "--out",
                "data/text/processed",
                "--val-ratio",
                "0.1",
            ]
        )

        run_command([args.python, "scripts/llm/train.py", "--config", args.pretrain_config])

    if not args.skip_sft:
        run_command([args.python, "scripts/llm/train_sft.py", "--config", args.sft_config])

    sft_checkpoint = checkpoint_from_config(args.sft_config)
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
