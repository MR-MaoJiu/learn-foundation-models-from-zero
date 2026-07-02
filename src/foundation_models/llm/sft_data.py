from __future__ import annotations

"""
SFT 数据处理。

推荐语料格式采用 OpenAI/GPT 常见 JSONL：

    {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]}

训练时会把 messages 渲染成 ChatML。最后一条 assistant 消息是监督目标，
前面的消息是上下文 prompt，并被 mask 成 `IGNORE_INDEX`，不参与 loss。
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch

from .chat import (
    CHATML_END,
    DEFAULT_SYSTEM_PROMPT,
    make_system_message,
    render_chat_messages,
)
from .tokenizer import LLMTokenizer


IGNORE_INDEX = -100
ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass
class SFTExample:
    """一条 SFT 样本。

    prompt：
        已经渲染好的 ChatML 上下文，末尾是 `<|assistant|>`。
    response：
        最后一条 assistant 的正文，后面会补 `<|end|>`，让模型学会停止。
    """

    prompt: str
    response: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def read_sft_jsonl(path: str | Path) -> list[SFTExample]:
    """读取 JSONL SFT 数据。"""

    examples: list[SFTExample] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            example = row_to_sft_example(row)
            if example is None:
                raise ValueError(f"Invalid SFT row at {path}:{line_number}")
            examples.append(example)

    if not examples:
        raise ValueError(f"No SFT examples found in {path}")
    return examples


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """把一行 JSON 标准化成 messages。

    正式流程只接受 OpenAI/GPT 风格 `messages`。
    这样可以避免旧的自由文本格式悄悄混入 SFT，导致训练和部署模板不一致。
    """

    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = _clean_text(message.get("role"))
        content = _clean_text(message.get("content"))
        if role not in ALLOWED_ROLES or not content:
            return None
        normalized.append({"role": role, "content": content})
    return normalized


def validate_messages(messages: list[dict[str, str]]) -> None:
    """检查 messages 是否适合 SFT。

    规则：
    - system 只能出现在第一条，且最多一条。
    - 去掉 system 后，角色必须 user/assistant 交替。
    - 最后一条必须是 assistant，因为它是当前样本的训练目标。
    """

    if not messages:
        raise ValueError("messages is empty")

    system_count = sum(1 for item in messages if item["role"] == "system")
    if system_count > 1:
        raise ValueError("messages has more than one system message")
    if system_count == 1 and messages[0]["role"] != "system":
        raise ValueError("system message must be the first message")

    dialogue = messages[1:] if messages[0]["role"] == "system" else messages
    if not dialogue:
        raise ValueError("messages has no user/assistant dialogue")
    if dialogue[0]["role"] != "user":
        raise ValueError("dialogue must start with a user message")
    if dialogue[-1]["role"] != "assistant":
        raise ValueError("SFT sample must end with an assistant message")

    expected = "user"
    for item in dialogue:
        if item["role"] != expected:
            raise ValueError(f"dialogue roles must alternate user/assistant; expected {expected}")
        expected = "assistant" if expected == "user" else "user"


def row_to_sft_example(row: dict[str, Any]) -> SFTExample | None:
    """把一行 JSON 转成 ChatML prompt/response。"""

    messages = row_to_messages(row)
    if messages is None:
        return None

    validate_messages(messages)

    # 没有 system 时自动补一个默认 system，和真实聊天产品的做法更接近。
    if messages[0]["role"] != "system":
        messages = [make_system_message(DEFAULT_SYSTEM_PROMPT)] + messages

    prompt_messages = messages[:-1]
    response_message = messages[-1]
    prompt = render_chat_messages(prompt_messages, add_generation_prompt=True)

    # response 里显式追加 <|end|>，让模型学会“这一轮助手回答结束”。
    # 训练脚本仍会额外 add_eos=True，表示整个样本结束。
    response = f"{response_message['content'].strip()}\n{CHATML_END}"
    return SFTExample(prompt=prompt, response=response)


def split_examples(
    examples: list[SFTExample],
    val_ratio: float,
    seed: int,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """按样本切分训练集和验证集。"""

    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, shuffled

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:] or shuffled
    return train_examples, val_examples


class ChatSFTDataset:
    """把 SFT 样本转成固定长度 batch。"""

    def __init__(
        self,
        examples: list[SFTExample],
        tokenizer: LLMTokenizer,
        block_size: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.examples)

    def encode_example(self, example: SFTExample) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(example.prompt, add_bos=True)
        response_ids = self.tokenizer.encode(example.response, add_eos=True)

        full_ids = prompt_ids + response_ids
        if len(full_ids) < 2:
            full_ids = full_ids + [self.tokenizer.eos_id]

        # 需要 block_size + 1 个 token 才能构造 input 和 next-token labels。
        full_ids = full_ids[: self.block_size + 1]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # label[i] 对应 full_ids[i + 1]。
        # 只有 target 位置进入 response 后，才参与 SFT loss。
        for index in range(len(labels)):
            target_position = index + 1
            if target_position < len(prompt_ids):
                labels[index] = IGNORE_INDEX

        pad_count = self.block_size - len(input_ids)
        if pad_count > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_count
            labels = labels + [IGNORE_INDEX] * pad_count

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self.examples), (batch_size,)).tolist()
        encoded = [self.encode_example(self.examples[index]) for index in indices]
        x = torch.stack([item[0] for item in encoded]).to(device=device)
        y = torch.stack([item[1] for item in encoded]).to(device=device)
        return x, y
