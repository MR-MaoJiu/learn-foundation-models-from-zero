from __future__ import annotations

"""
Chat template and inference helpers.

External data uses OpenAI/GPT-style `messages` dictionaries:

    {"role": "system", "content": "..."}
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}

Before the text enters the model, messages are rendered into a compact ChatML
format with explicit role tokens. This keeps training, evaluation, generation,
and deployment on the same prompt format.
"""

from pathlib import Path
from typing import Any

import torch

from .config import model_config_from_dict
from .model import GPT
from .tokenizer import LLMTokenizer


CHATML_SYSTEM = "<|system|>"
CHATML_USER = "<|user|>"
CHATML_ASSISTANT = "<|assistant|>"
CHATML_END = "<|end|>"

DEFAULT_SYSTEM_PROMPT = "你是一个中文 AI 助手。回答要准确、自然、简洁；不知道实时信息时要说明限制。"

DEFAULT_CHAT_STOPS = (
    CHATML_END,
    CHATML_USER,
    CHATML_SYSTEM,
    "\n" + CHATML_USER,
    "\n" + CHATML_SYSTEM,
    "<eos>",
)


def make_system_message(content: str = DEFAULT_SYSTEM_PROMPT) -> dict[str, str]:
    """Create a system message."""

    return {"role": "system", "content": content}


def make_user_message(content: str) -> dict[str, str]:
    """Create a user message."""

    return {"role": "user", "content": content}


def make_assistant_message(content: str) -> dict[str, str]:
    """Create an assistant message."""

    return {"role": "assistant", "content": content}


def role_to_token(role: str) -> str:
    """Map a `messages` role to its ChatML token."""

    if role == "system":
        return CHATML_SYSTEM
    if role == "user":
        return CHATML_USER
    if role == "assistant":
        return CHATML_ASSISTANT
    raise ValueError(f"Unsupported chat role: {role}")


def render_chat_messages(
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render OpenAI/GPT-style messages into ChatML text."""

    parts: list[str] = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        if not content:
            continue
        parts.append(f"{role_to_token(role)}\n{content}\n{CHATML_END}")

    if add_generation_prompt:
        parts.append(CHATML_ASSISTANT)

    return "\n".join(parts)


def build_chat_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Wrap a plain user prompt as a ChatML generation prompt."""

    messages = [make_system_message(system_prompt), make_user_message(prompt.strip())]
    return render_chat_messages(messages, add_generation_prompt=True)


def trim_assistant_reply(text: str, prompt_text: str) -> str:
    """Extract the current assistant turn from the full generated text."""

    if text.startswith(prompt_text):
        reply = text[len(prompt_text) :]
    else:
        assistant_index = text.rfind(CHATML_ASSISTANT)
        if assistant_index >= 0:
            reply = text[assistant_index + len(CHATML_ASSISTANT) :]
        else:
            reply = text

    for stop in DEFAULT_CHAT_STOPS:
        stop_index = reply.find(stop)
        if stop_index >= 0:
            reply = reply[:stop_index]

    return reply.strip()


def load_chat_model(
    checkpoint_path: str | Path,
    device: torch.device,
    tokenizer_path: str | Path | None = None,
) -> tuple[GPT, LLMTokenizer, dict[str, Any]]:
    """Load checkpoint, tokenizer, and model."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint["config"]

    tokenizer_source = tokenizer_path or raw_config["training"]["tokenizer_path"]
    tokenizer = LLMTokenizer(tokenizer_source)

    model_config = model_config_from_dict(raw_config["model"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, raw_config


def chat_stop_token_ids(tokenizer: LLMTokenizer) -> set[int]:
    """Return token ids that should stop current-turn chat generation."""

    ids = {tokenizer.eos_id}
    if tokenizer.chat_end_id is not None:
        ids.add(tokenizer.chat_end_id)
    return ids


def generate_text(
    model: GPT,
    tokenizer: LLMTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    stop_token_ids: set[int] | None = None,
) -> str:
    """Run autoregressive generation and return the full decoded text."""

    input_ids = tokenizer.encode(prompt_text, add_bos=True)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids=x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
        )

    text = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
    if text.startswith("<bos>"):
        text = text[len("<bos>") :].lstrip()
    return text


def generate_chat_reply(
    model: GPT,
    tokenizer: LLMTokenizer,
    user_text: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    history: str = "",
) -> tuple[str, str]:
    """Generate one assistant reply and return `(reply, new_history)`."""

    if history:
        prompt_text = f"{history}\n{CHATML_USER}\n{user_text.strip()}\n{CHATML_END}\n{CHATML_ASSISTANT}"
    else:
        prompt_text = build_chat_prompt(user_text)

    full_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=chat_stop_token_ids(tokenizer),
    )
    reply = trim_assistant_reply(full_text, prompt_text)
    new_history = f"{prompt_text}\n{reply}\n{CHATML_END}".strip()
    return reply, new_history
