from __future__ import annotations

"""
SFT data utilities.

The recommended corpus format is OpenAI/GPT-style JSONL:

    {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]}

During training, each row is rendered into ChatML. All prompt tokens are
masked with `IGNORE_INDEX`, so loss is applied only to the assistant answer.
That is the key difference between "continue this text" and "answer the user".
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
    """A single rendered SFT example.

    prompt:
        ChatML context ending with `<|assistant|>`.
    response:
        The final assistant answer plus `<|end|>`.
    """

    prompt: str
    response: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def read_sft_jsonl(path: str | Path) -> list[SFTExample]:
    """Read and validate a JSONL SFT file."""

    examples: list[SFTExample] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            example = row_to_sft_example(item)
            if example is None:
                raise ValueError(f"Invalid SFT row at {path}:{line_number}")
            examples.append(example)

    if not examples:
        raise ValueError(f"No SFT examples found in {path}")
    return examples


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Normalize one JSON row into a `messages` list.

    The formal workflow accepts only structured `messages`. This prevents old
    free-form transcripts such as "用户：... 助手：..." from silently mixing
    into SFT and teaching the model to keep writing both sides.
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
    """Validate whether a `messages` list is suitable for chat SFT."""

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
    """Convert one JSON row into ChatML prompt and assistant response."""

    messages = row_to_messages(row)
    if messages is None:
        return None

    validate_messages(messages)

    # If a row has no system message, add the deployment default. This mirrors
    # how many chat products attach a default system instruction at runtime.
    if messages[0]["role"] != "system":
        messages = [make_system_message(DEFAULT_SYSTEM_PROMPT)] + messages

    prompt_messages = messages[:-1]
    response_message = messages[-1]
    prompt = render_chat_messages(prompt_messages, add_generation_prompt=True)

    # The answer explicitly learns `<|end|>`, so generation can stop at the
    # end of the current assistant turn instead of continuing with a fake user.
    response = f"{response_message['content'].strip()}\n{CHATML_END}"
    return SFTExample(prompt=prompt, response=response)


def split_examples(
    examples: list[SFTExample],
    val_ratio: float,
    seed: int,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Split examples into train and validation sets."""

    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, shuffled

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:] or shuffled
    return train_examples, val_examples


class ChatSFTDataset:
    """Turn SFT examples into fixed-size training batches."""

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

        # Need block_size + 1 tokens to build input ids and next-token labels.
        full_ids = full_ids[: self.block_size + 1]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # labels[i] corresponds to full_ids[i + 1]. Only targets inside the
        # assistant response should contribute to SFT loss.
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
