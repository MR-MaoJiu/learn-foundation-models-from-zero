from __future__ import annotations

"""
数据处理模块。

语言模型不能直接吃中文字符串，它只能吃数字。
所以数据处理分成两层：

1. tokenizer 把文本变成 token id。
   例如：
       "语言模型" -> [120, 88, 341]

2. Dataset 从 token id 长序列里切出训练样本。
   例如一段 token：
       [10, 20, 30, 40, 50]
   可以构造成：
       x = [10, 20, 30, 40]
       y = [20, 30, 40, 50]

模型看到 x，要预测 y。
这就是“下一个 token 预测”。
"""

from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch

from .tokenizer import LLMTokenizer


def read_text_files(paths: Iterable[str | Path]) -> list[str]:
    """读取文本文件，并切成多个文档段落。

    参数：
    - paths：一个或多个 `.txt` 文件路径。

    返回：
    - documents：字符串列表，每个元素是一小段文本。

    为什么按空行切？
    - 空行通常表示一个自然段结束。
    - 我们可以把每个段落当成一个小文档。
    - 后面会在每个文档末尾加 `<eos>`，告诉模型“这里结束了”。
    """

    documents: list[str] = []
    for path in paths:
        path = Path(path)

        # encoding="utf-8" 对中文很重要。
        # 如果编码不对，中文会变成乱码。
        text = path.read_text(encoding="utf-8")

        # "\n\n" 表示空行。
        # strip() 去掉段落开头结尾的空白字符。
        for part in text.split("\n\n"):
            part = part.strip()
            if part:
                documents.append(part)
    return documents


def tokenize_documents(tokenizer: LLMTokenizer, documents: list[str]) -> list[int]:
    """把很多段文本变成一条连续 token id 流。

    为什么最后要 add_eos=True？
    - eos 是 end of sequence，表示一段文本结束。
    - 如果不加 eos，两个完全无关的段落会硬接在一起。
    - 加 eos 后，模型能学到“这里可以停止/换文档”。
    """

    all_ids: list[int] = []
    for doc in documents:
        all_ids.extend(tokenizer.encode(doc, add_eos=True))
    return all_ids


def save_token_array(ids: list[int], path: str | Path, vocab_size: int) -> None:
    """把 token id 保存成二进制文件。

    为什么不用普通 txt 保存？
    - txt 可读性好，但体积大，读取慢。
    - 训练时数据会被频繁读取，二进制更高效。

    为什么有 uint16 / uint32？
    - 如果词表小于 65536，uint16 就够了，每个 token 只占 2 字节。
    - 如果词表更大，要用 uint32，每个 token 占 4 字节。
    """

    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    array = np.array(ids, dtype=dtype)

    # 确保输出目录存在。
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # tofile 会直接写原始二进制数据。
    array.tofile(path)


def make_train_val_bins(
    input_files: Iterable[str | Path],
    tokenizer_path: str | Path,
    out_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path]:
    """从原始文本生成 `train.bin` 和 `val.bin`。

    train.bin：
    - 训练集。
    - 用来计算 loss，并更新模型参数。

    val.bin：
    - 验证集。
    - 只用来看模型表现，不更新参数。

    为什么需要验证集？
    - 如果模型只在训练集上越来越好，可能只是背下来了。
    - 验证集能帮助我们观察模型有没有学到更通用的规律。
    """

    tokenizer = LLMTokenizer(tokenizer_path)
    documents = read_text_files(input_files)

    # 打乱文档顺序。
    # seed 固定后，每次打乱结果基本一致，方便复现实验。
    random.Random(seed).shuffle(documents)

    # 所有文档变成一条 token 长序列。
    all_ids = tokenize_documents(tokenizer, documents)

    # 按 token 数切分验证集。
    # 对小语料来说，按 token 切比按文档切更稳。
    val_count = max(1, int(len(all_ids) * val_ratio))
    val_ids = all_ids[:val_count]
    train_ids = all_ids[val_count:]

    # 极小语料兜底，避免文件为空。
    if not train_ids:
        train_ids = all_ids
    if not val_ids:
        val_ids = all_ids

    out_dir = Path(out_dir)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    save_token_array(train_ids, train_path, tokenizer.vocab_size)
    save_token_array(val_ids, val_path, tokenizer.vocab_size)
    return train_path, val_path


class BinaryTokenDataset:
    """从 `.bin` token 文件中随机取训练 batch。

    这个类不是 PyTorch 标准 Dataset 的写法，而是更接近很多 LLM 训练代码的做法：
    - 先把所有 token 存成一个大数组。
    - 每次随机选一个起点。
    - 从这个起点切出 block_size + 1 个 token。
    - 前 block_size 个作为 x，后 block_size 个作为 y。
    """

    def __init__(self, path: str | Path, block_size: int, vocab_size: int):
        self.path = Path(path)
        self.block_size = block_size
        self.dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        # np.memmap 不会一次性把整个文件读进内存。
        # 它像“映射”一样，需要哪一段就读哪一段。
        # 大语料训练时这很重要。
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")

        # 需要至少 block_size + 1 个 token。
        # 因为 x 要 block_size 个，y 是向后移动一位，也需要最后那个答案 token。
        if len(self.tokens) <= block_size + 1:
            raise ValueError(
                f"{self.path} has {len(self.tokens)} tokens, but block_size is {block_size}. "
                "Use more data or reduce max_seq_len in the config."
            )

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """随机生成一个训练 batch。

        返回：
        - x：输入 token，形状 [batch_size, block_size]
        - y：答案 token，形状 [batch_size, block_size]

        举例：
            chunk = [10, 20, 30, 40, 50]
            x     = [10, 20, 30, 40]
            y     = [20, 30, 40, 50]

        这样模型在每个位置都做“预测下一个 token”。
        """

        # 最大起点。
        # 起点不能太靠后，否则切不出 block_size + 1 个 token。
        max_start = len(self.tokens) - self.block_size - 1

        # 随机生成 batch_size 个起点。
        starts = torch.randint(0, max_start, (batch_size,))

        x_list = []
        y_list = []
        for start in starts.tolist():
            # 取 block_size + 1 个 token。
            chunk = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)

            # 前面部分作为输入。
            x_list.append(torch.from_numpy(chunk[:-1]))

            # 后面移动一位作为答案。
            y_list.append(torch.from_numpy(chunk[1:]))

        # 把 list 叠成一个 Tensor。
        x = torch.stack(x_list).to(device=device, dtype=torch.long)
        y = torch.stack(y_list).to(device=device, dtype=torch.long)
        return x, y
