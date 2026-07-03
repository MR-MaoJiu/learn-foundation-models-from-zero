from __future__ import annotations

"""
Build a larger plain-text corpus for base pretraining.

Pretraining and SFT use different data shapes:
- Pretraining wants ordinary text paragraphs and learns next-token prediction.
- SFT wants structured `messages` JSONL and learns assistant behavior.

The tiny seed file in this repository is intentionally small, so it is not
enough for `max_seq_len=1024`. This script creates a larger reproducible
Chinese corpus by combining:
1. Existing seed `.txt` files.
2. Chat rows converted into readable dialogue transcripts.
3. Synthetic educational and daily-life paragraphs.

The output is plain UTF-8 text. It is a teaching corpus, not a replacement for
licensed large-scale real-world pretraining data.
"""

import argparse
import json
from pathlib import Path
import random
from typing import Any


TOPICS = [
    "语言模型会根据上下文预测下一个 token。这个目标看起来简单，但只要数据足够多，模型就能学到语法、常识、格式和一部分推理模式。",
    "训练 tokenizer 时要固定语料和词表。后续预训练、SFT、评估和部署都必须使用同一个 tokenizer，否则 checkpoint 的词表大小会不匹配。",
    "预训练阶段的目标是学习通用文本分布。它更像是在学习如何续写自然文本，而不是学习如何扮演聊天助手。",
    "SFT 阶段使用 messages 数据，让模型看到系统指令、用户问题和助手回答之间的边界。这样模型更容易学会只回答当前问题。",
    "验证集不参与参数更新。它的作用是观察模型对未见数据的表现，帮助判断训练是否过拟合。",
    "如果训练 loss 下降而验证 loss 上升，通常说明模型开始记忆训练集。可以减少训练步数、增加数据、调低模型容量或加强正则化。",
    "部署模型前需要做烟雾评估。至少要检查回答是否为空、是否异常重复、是否泄露模板 token、是否能在合理长度内停止。",
    "真实项目会记录每次训练的数据版本、配置、随机种子、代码提交、指标和产物路径。这样后续才能复现和排查问题。",
    "小模型的能力有限。它适合学习完整流程和验证工程链路，不应该被当作商业级问答系统使用。",
    "当模型回答实时天气、医疗、法律或金融问题时，如果没有可靠实时来源，就应该说明限制，而不是编造确定答案。",
    "工程排查通常先看输入和输出，再看配置和依赖。训练问题也一样，先确认数据格式、tokenizer、checkpoint 和设备是否一致。",
    "一个好的 README 应该说明目标、安装方式、最短运行路径、分步命令、数据格式、产物目录和常见问题。",
    "学习复杂系统时，先跑通最小闭环很重要。小闭环包括数据准备、训练、评估、导出和一次真实请求。",
    "模型生成重复内容时，可以检查采样参数、训练数据重复度和停止条件。小模型尤其容易在高温度下发散。",
    "如果 SFT 后模型仍然继续写用户问题，通常要检查模板和 label mask，确保只有 assistant 回复参与监督。",
]


DAILY_PARAGRAPHS = [
    "早上出门前可以先看天气、交通和日程。把证件、钥匙、手机和充电器放在固定位置，能减少临时寻找的时间。",
    "做饭时先确定主菜，再搭配一个简单蔬菜。对于忙碌的晚上，番茄鸡蛋、青菜豆腐和简单汤类都是稳定选择。",
    "学习计划不需要一开始就很复杂。每天固定一个小目标，例如读一节文档、写一个函数或复盘一个错误，比偶尔长时间学习更容易坚持。",
    "沟通时先说共同目标，再说具体请求。这样对方更容易理解你的动机，也更容易一起解决问题。",
    "整理任务列表时，可以把任务分成必须完成、应该完成和可以延后。先处理阻塞后续进度的事项。",
    "写作时先确定读者是谁，再决定语气和结构。技术文档需要清楚、准确、可执行；日常消息需要自然、礼貌、简洁。",
    "遇到焦虑时，可以先把问题写下来，区分事实、猜测和下一步行动。很多压力来自把未知情况混在一起。",
    "复盘不是为了责备自己，而是为了找出下一次可以改进的一处细节。好的复盘会包含现象、原因、影响和行动项。",
]


def read_seed_texts(paths: list[str]) -> list[str]:
    paragraphs: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for part in text.split("\n\n"):
            part = part.strip()
            if part:
                paragraphs.append(part)
    return paragraphs


def messages_to_dialogue(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            return None
        role_name = {"system": "系统", "user": "用户", "assistant": "助手"}.get(role, role)
        lines.append(f"{role_name}：{content}")
    return "\n".join(lines)


def read_sft_as_text(paths: list[str], max_rows: int) -> list[str]:
    paragraphs: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(paragraphs) >= max_rows:
                    return paragraphs
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dialogue = messages_to_dialogue(item)
                if dialogue:
                    paragraphs.append(dialogue)
    return paragraphs


def synthetic_paragraphs(target_count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    openers = [
        "从工程角度看，",
        "对于初学者来说，",
        "在真实项目里，",
        "如果要稳定复现结果，",
        "排查问题时，",
        "写文档时，",
        "部署前，",
        "训练小模型时，",
    ]
    actions = [
        "先确认输入数据是否干净，再检查配置是否一致。",
        "先跑通一个最小样例，再逐步增加数据规模和训练步数。",
        "要把产物目录、配置文件和模型权重分清楚，避免混用旧文件。",
        "要记录关键指标，例如训练 loss、验证 loss、样本数量和词表大小。",
        "要优先修复会阻塞主流程的问题，再处理体验优化。",
        "要让每一步都有可检查的输出，失败时才能快速定位。",
        "要使用统一的聊天模板，让训练和推理看到相同的角色边界。",
        "要给模型明确的停止标记，避免它继续生成下一轮用户内容。",
    ]

    paragraphs: list[str] = []
    while len(paragraphs) < target_count:
        topic = rng.choice(TOPICS)
        daily = rng.choice(DAILY_PARAGRAPHS)
        paragraph = (
            f"{rng.choice(openers)}{rng.choice(actions)}{topic}"
            f" 举个日常类比：{daily}"
        )
        paragraphs.append(paragraph)
    return paragraphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a larger Chinese plain-text pretraining corpus.")
    parser.add_argument("--out", default="data/text/raw/pretrain_zh.txt")
    parser.add_argument("--seed-text", nargs="*", default=["data/text/raw/tiny_zh_corpus.txt"])
    parser.add_argument("--sft-jsonl", nargs="*", default=["data/text/raw/sft_chat_zh.jsonl"])
    parser.add_argument("--count", type=int, default=8000, help="Number of synthetic paragraphs to add.")
    parser.add_argument("--max-sft-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paragraphs = []
    paragraphs.extend(read_seed_texts(args.seed_text))
    paragraphs.extend(read_sft_as_text(args.sft_jsonl, max_rows=args.max_sft_rows))
    paragraphs.extend(synthetic_paragraphs(args.count, seed=args.seed))

    # Stable de-duplication keeps the corpus deterministic.
    seen: set[str] = set()
    unique: list[str] = []
    for paragraph in paragraphs:
        if paragraph not in seen:
            seen.add(paragraph)
            unique.append(paragraph)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(unique) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(unique)} pretraining paragraphs to {out_path}")


if __name__ == "__main__":
    main()
