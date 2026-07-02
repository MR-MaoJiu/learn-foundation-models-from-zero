from __future__ import annotations

"""
导出部署包。

训练 checkpoint 往往散落在 checkpoints/ 下。
部署时更希望得到一个完整目录：

    deployments/llm/chat_debug/
    ├─ model.pt
    ├─ tokenizer.json
    └─ manifest.json
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint into a deployable model directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--out", required=True, help="Output deployment directory.")
    parser.add_argument("--name", default="educational-llm-chat", help="Model package name.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    tokenizer_path = Path(config["training"]["tokenizer_path"])

    model_out = out_dir / "model.pt"
    tokenizer_out = out_dir / "tokenizer.json"
    shutil.copy2(checkpoint_path, model_out)
    shutil.copy2(tokenizer_path, tokenizer_out)

    manifest = {
        "name": args.name,
        "format": "pytorch_checkpoint",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint_path),
        "model_file": "model.pt",
        "tokenizer_file": "tokenizer.json",
        "step": checkpoint.get("step"),
        "model_config": config["model"],
        "generation_defaults": {
            "max_new_tokens": 80,
            "temperature": 0.3,
            "top_k": 20,
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported model package: {out_dir}")
    print(f"- {model_out}")
    print(f"- {tokenizer_out}")
    print(f"- {manifest_path}")


if __name__ == "__main__":
    main()
