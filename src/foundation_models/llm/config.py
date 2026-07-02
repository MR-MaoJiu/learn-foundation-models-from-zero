from __future__ import annotations

"""
配置模块。

训练 LLM 会有很多超参数：
- 模型有多少层？
- 每个 token 向量多少维？
- 注意力头数是多少？
- 上下文长度是多少？

这些设置如果散落在代码里，会很难改。
所以我们把模型结构参数集中放在 ModelConfig 里。
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """构建 Transformer 需要的模型超参数。

    dataclass 是 Python 提供的“数据类”。
    它适合保存一组配置，不需要我们手写很多 __init__ 代码。
    """

    # 词表大小。
    # 如果 vocab_size=8000，模型每个位置会输出 8000 个 token 分数。
    vocab_size: int = 32768

    # 最大上下文长度。
    # max_seq_len=128 表示模型一次最多看 128 个 token。
    max_seq_len: int = 1024

    # Transformer Block 层数。
    # 层数越多，模型越深，能力通常越强，也越慢。
    n_layer: int = 30

    # 注意力头数。
    # 多头注意力允许模型从多个角度关注上下文。
    n_head: int = 12

    # 隐藏维度。
    # 每个 token 会被表示成 n_embd 维向量。
    n_embd: int = 768

    # MLP 中间层维度。
    # 通常会比 n_embd 大，用来增强表达能力。
    intermediate_size: int = 3072

    # dropout 概率。
    # 训练时随机丢掉一部分信息，帮助降低过拟合。
    dropout: float = 0.0

    # RoPE 的频率基数。
    # 初学阶段不用调它，知道它和位置编码有关即可。
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        """创建配置后自动检查是否合法。

        注意力会把 hidden 维度平均分给多个头：
            head_dim = n_embd / n_head

        所以 n_embd 必须能被 n_head 整除。
        """

        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        """每个注意力头拿到的向量维度。"""

        return self.n_embd // self.n_head


def load_json_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置文件。

    JSON 的好处是直观，初学者打开就能看到所有训练参数。
    """

    path = Path(path)
    # utf-8-sig 既能读取普通 UTF-8，也能兼容带 BOM 的 UTF-8 文件。
    # 有些编辑器或系统工具保存 JSON 时会带 BOM。
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def model_config_from_dict(values: dict[str, Any]) -> ModelConfig:
    """把字典转换成 ModelConfig。

    这里会检查有没有写错配置名。
    例如你把 `n_layer` 写成 `num_layer`，这里会直接报错。
    早报错比训练到一半才出问题更好。
    """

    allowed = set(ModelConfig.__dataclass_fields__.keys())
    unknown = set(values.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
    return ModelConfig(**values)
