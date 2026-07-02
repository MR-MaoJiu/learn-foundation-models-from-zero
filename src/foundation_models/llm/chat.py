from __future__ import annotations

"""
聊天模板和推理工具。

本项目内部使用一个 ChatML 风格模板来模拟主流聊天模型的输入格式。
外部数据仍然使用 OpenAI/GPT 常见的 `messages` JSON 结构：

    {"role": "system", "content": "..."}
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}

进入模型前会被渲染成：

    <|system|>
    你是一个...
    <|end|>
    <|user|>
    问题
    <|end|>
    <|assistant|>

这样做比直接写“用户：/助手：”更接近主流模型训练方式：

1. 角色边界稳定，不容易和正文混淆。
2. tokenizer 可以把角色标记注册成单独 special token。
3. 生成时遇到 <|end|> 就能停止当前轮回复。
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

# 生成时如果模型开始输出下一条角色 token，说明当前助手轮次应该结束。
DEFAULT_CHAT_STOPS = (
    CHATML_END,
    CHATML_USER,
    CHATML_SYSTEM,
    "\n" + CHATML_USER,
    "\n" + CHATML_SYSTEM,
    "<eos>",
)


def make_system_message(content: str = DEFAULT_SYSTEM_PROMPT) -> dict[str, str]:
    """创建 system 消息。

    system 消息告诉模型“你是谁、应该怎么回答”。真实聊天模型通常都会有
    system/developer 层级的指令；教学项目里先保留一个简单 system。
    """

    return {"role": "system", "content": content}


def make_user_message(content: str) -> dict[str, str]:
    """创建 user 消息。"""

    return {"role": "user", "content": content}


def make_assistant_message(content: str) -> dict[str, str]:
    """创建 assistant 消息。"""

    return {"role": "assistant", "content": content}


def role_to_token(role: str) -> str:
    """把 messages 里的 role 转成 ChatML token。"""

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
    """把 OpenAI/GPT 风格 messages 渲染成模型输入文本。

    参数：
    - messages：形如 [{"role": "user", "content": "..."}] 的消息列表。
    - add_generation_prompt：是否在末尾追加 `<|assistant|>`。

    SFT 构造 prompt 时会打开 `add_generation_prompt`，让模型从 assistant
    角色开始生成；构造完整训练文本时 assistant 回复内容会放在它后面。
    """

    parts: list[str] = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        if not content:
            continue

        # 每条消息都用“角色 token + 内容 + <|end|>”包起来。
        # <|end|> 是明确的消息边界，生成时也可以作为停止标记。
        parts.append(f"{role_to_token(role)}\n{content}\n{CHATML_END}")

    if add_generation_prompt:
        parts.append(CHATML_ASSISTANT)

    return "\n".join(parts)


def build_chat_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """把普通用户输入整理成 ChatML 推理 prompt。"""

    prompt = prompt.strip()
    messages = [make_system_message(system_prompt), make_user_message(prompt)]
    return render_chat_messages(messages, add_generation_prompt=True)


def trim_assistant_reply(text: str, prompt_text: str) -> str:
    """从完整生成文本中截取助手当前这一轮回复。"""

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
    """加载 checkpoint、tokenizer 和模型。"""

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
    """返回聊天生成时应该提前停止的 token id。"""

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
    """执行一次自回归生成，并返回完整文本。"""

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

    # 生成时我们在最前面加了 <bos>，但 ChatML prompt 本身不包含它。
    # 去掉它以后，`trim_assistant_reply` 才能直接用 prompt 前缀截断。
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
    """生成一轮助手回复，并返回 `(reply, new_history)`。

    `history` 是内部 ChatML 文本。HTTP 服务会把它回传给调用方，下一轮请求
    再带回来，就能模拟多轮对话。
    """

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
