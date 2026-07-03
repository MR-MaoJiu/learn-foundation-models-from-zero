from __future__ import annotations

"""
Smoke-test a chat checkpoint.

This is intentionally small. A real project would add larger benchmark sets,
manual review, factuality checks, safety checks, latency metrics, and cost
tracking. Here we only verify that the model can answer several common prompts
without empty output or obvious role-token leakage.
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.chat import CHATML_SYSTEM, CHATML_USER, generate_chat_reply, load_chat_model
from foundation_models.llm.utils import choose_device


DEFAULT_PROMPTS = [
    "今天天气怎么样？",
    "你能解释一下语言模型是什么吗？",
    "为什么我的模型会一直续写用户和助手？",
    "训练 tokenizer 的作用是什么？",
    "部署前需要检查什么？",
]


def load_prompts(path: str | Path | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS

    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chat checkpoint with smoke-test prompts.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--prompts", default="", help="Optional txt file, one prompt per line.")
    parser.add_argument("--out", default="runs/llm/chat_eval.jsonl", help="Where to write JSONL results.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, tokenizer, _ = load_chat_model(args.checkpoint, device)
    prompts = load_prompts(args.prompts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with out_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            reply, _ = generate_chat_reply(
                model=model,
                tokenizer=tokenizer,
                user_text=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            checks = {
                "non_empty": bool(reply.strip()),
                "no_role_token": CHATML_USER not in reply and CHATML_SYSTEM not in reply,
                "not_too_long": len(reply) <= 300,
            }
            passed = all(checks.values())
            failures += 0 if passed else 1

            item = {
                "prompt": prompt,
                "reply": reply,
                "checks": checks,
                "passed": passed,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            status = "PASS" if passed else "FAIL"
            print(f"[{status}] 用户：{prompt}")
            print(f"助手：{reply}")
            print()

    print(f"Evaluated {len(prompts)} prompts. Failures: {failures}. Results: {out_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
