from __future__ import annotations

"""多模态模块配置。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class MultimodalConfig:
    """TinyCLIP 需要的参数。"""

    image_channels: int = 1
    vocab_size: int = 128
    text_embed_dim: int = 64
    embed_dim: int = 64
    hidden_channels: int = 32
    max_text_len: int = 32


def load_json_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def multimodal_config_from_dict(values: dict[str, Any]) -> MultimodalConfig:
    allowed = set(MultimodalConfig.__dataclass_fields__.keys())
    unknown = set(values.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown multimodal config keys: {sorted(unknown)}")
    return MultimodalConfig(**values)
