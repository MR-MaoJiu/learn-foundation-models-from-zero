from __future__ import annotations

"""
第四步脚本：用训练好的 checkpoint 生成文本。

训练阶段做的是：

    输入 x -> 模型预测 -> 计算 loss -> 更新参数

生成阶段做的是：

    prompt 文本
    -> tokenizer 编码成 token id
    -> 模型预测下一个 token
    -> 把新 token 接到后面
    -> 继续预测下一个 token
    -> 最后 decode 回中文

注意：
模型不是一次性生成一整段话，而是“一次生成一个 token”。
这就是自回归生成。
"""

import argparse
from pathlib import Path
import sys


# 找到项目根目录，并把 src 加入 Python 导入路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.chat import (
    build_chat_prompt,
    chat_stop_token_ids,
    generate_chat_reply,
    generate_text,
    load_chat_model,
    trim_assistant_reply,
)
from foundation_models.llm.utils import choose_device


def main() -> None:
    """命令行入口。

    示例：

        python scripts/llm/generate.py --checkpoint checkpoints/llm/debug/last.pt --prompt "语言模型的目标是"
    """

    # 有些终端默认编码不一定是 UTF-8。
    # 模型早期生成的字符可能比较乱，如果终端编码不支持就会报错。
    # 这里把输出改成 UTF-8，并用 replace 替换无法显示的字符。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")

    # checkpoint 是训练保存下来的 .pt 文件。
    # 它里面包含模型参数和配置。
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint.",
    )

    # prompt 是你给模型的开头文本。
    # 模型会接着它往后写。
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt text.",
    )

    # 默认用聊天模式：把普通输入包装成 ChatML，并且只显示助手回复。
    # 如果你想保留原来的纯续写行为，可以使用 --mode completion。
    parser.add_argument(
        "--mode",
        choices=["chat", "completion"],
        default="chat",
        help="chat wraps the prompt as a user turn; completion continues raw text.",
    )

    # 连续对话模式。
    # 开启后，脚本会保留本轮终端中的历史上下文。
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive chat loop.",
    )

    # 调试时可以打开，查看真正喂给模型和模型完整续写出的内容。
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Print the full generated text instead of only the assistant reply in chat mode.",
    )

    # 最多生成多少个新 token。
    # 注意 token 不等于汉字，一个汉字可能是一个 token，也可能被拆成多个 token。
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )

    # temperature 控制随机性。
    # 低：更保守。
    # 高：更随机。
    # 0 或更低：直接选最高分 token。
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    # top_k 表示每次只从分数最高的 k 个 token 中选择。
    # 它可以减少非常离谱的随机输出。
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    # device="auto" 时，有 GPU 就用 GPU，否则用 CPU。
    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    # 选择 CPU 或 GPU。
    device = choose_device(args.device)

    # 加载 checkpoint、tokenizer 和模型。
    # 必须使用训练时同一个 tokenizer，否则 token id 对不上。
    model, tokenizer, _ = load_chat_model(args.checkpoint, device)

    if args.interactive:
        history = ""
        print("进入对话模式。输入 exit 或 quit 结束。")
        while True:
            try:
                user_text = input("用户：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            reply, history = generate_chat_reply(
                model=model,
                tokenizer=tokenizer,
                user_text=user_text,
                history=history,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"助手：{reply}")
        return

    if args.mode == "chat":
        prompt_text = build_chat_prompt(args.prompt)
    else:
        prompt_text = args.prompt

    full_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_token_ids=chat_stop_token_ids(tokenizer) if args.mode == "chat" else None,
    )

    if args.mode == "chat" and not args.show_full:
        print(trim_assistant_reply(full_text, prompt_text))
    else:
        # completion 模式保留原来的行为：输出完整文本。
        print(full_text)


if __name__ == "__main__":
    main()
