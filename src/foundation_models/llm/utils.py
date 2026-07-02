from __future__ import annotations

"""
工具函数模块。

这里放一些不属于模型结构、也不属于数据处理的小工具：
- 固定随机种子
- 选择 CPU/GPU
- 选择训练精度
- 计算学习率
- 保存 checkpoint
"""

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，让实验更容易复现。

    训练里有很多随机性：
    - 参数随机初始化
    - 随机抽 batch
    - dropout 随机丢弃

    固定 seed 后，同一台机器上多次运行会更接近。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    """选择训练设备。

    name="auto" 时：
    - 有 NVIDIA CUDA GPU 就用 cuda。
    - 否则用 cpu。
    """

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    """选择数字精度。

    常见精度：
    - fp32：最稳，但占显存多。
    - fp16：省显存，GPU 上常用，但数值范围小。
    - bf16：也省显存，数值范围比 fp16 友好，但需要硬件支持。

    初学可以不用手动改，配置里保持 "auto" 即可。
    """

    if name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
) -> float:
    """计算当前 step 的学习率。

    学习率控制每次参数更新的步子大小。

    这个函数分两段：

    1. warmup 阶段：
       学习率从很小慢慢升高。
       这样训练一开始不容易炸。

    2. cosine decay 阶段：
       学习率按余弦曲线慢慢降低。
       后期步子变小，有利于模型收敛。
    """

    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (learning_rate - min_learning_rate)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> None:
    """保存 checkpoint。

    checkpoint 里保存：
    - model.state_dict()：模型参数。
    - optimizer.state_dict()：优化器状态。
    - step：训练到了第几步。
    - config：当时使用的配置。

    之后生成文本只需要模型参数和配置。
    如果要断点续训，还需要优化器状态。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )
