from __future__ import annotations

"""视觉模块配置。

这里先用 MNIST 手写数字分类做入门任务。

为什么从分类开始？
- 图像分类比图像生成、多模态对齐更容易理解。
- 它能让你先学会图像如何变成张量。
- 也能理解卷积网络如何从图像里提取特征。
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class VisionConfig:
    """构建小型 CNN 需要的模型参数。"""

    # MNIST 是 10 类：数字 0 到 9。
    num_classes: int = 10

    # MNIST 图片大小是 28x28。
    image_size: int = 28

    # MNIST 是灰度图，所以通道数是 1。
    # RGB 彩色图通道数通常是 3。
    channels: int = 1

    # 卷积层里使用多少个特征通道。
    # 越大模型越强，也越慢。
    hidden_channels: int = 32

    # dropout 用来降低过拟合。
    dropout: float = 0.1


def load_json_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置文件，兼容普通 UTF-8 和带 BOM 的 UTF-8。"""

    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def vision_config_from_dict(values: dict[str, Any]) -> VisionConfig:
    """把字典转换成 VisionConfig，并检查是否有拼错字段。"""

    allowed = set(VisionConfig.__dataclass_fields__.keys())
    unknown = set(values.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown vision config keys: {sorted(unknown)}")
    return VisionConfig(**values)
