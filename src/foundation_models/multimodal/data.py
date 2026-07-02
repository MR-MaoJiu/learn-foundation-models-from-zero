from __future__ import annotations

"""多模态数据读取。

数据文件是 JSONL，每一行类似：

{"image": "data/vision/mnist/train/images/000001.png", "text": "这是一张数字 7 的手写图片。"}

训练时返回：
- 图像张量
- 文本 token id
- 文本 mask
"""

from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch


class CharTokenizer:
    """极简字符 tokenizer。

    这个 tokenizer 只为了教学，不适合真实大规模文本。
    """

    def __init__(self, max_len: int):
        self.max_len = max_len

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = []
        for ch in text[: self.max_len]:
            # 把字符编码压到 1..127，0 留给 padding。
            ids.append((ord(ch) % 127) + 1)

        mask = [1] * len(ids)
        while len(ids) < self.max_len:
            ids.append(0)
            mask.append(0)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


class ImageTextDataset:
    """图文配对数据集。"""

    def __init__(self, jsonl_path: str | Path, max_text_len: int):
        self.path = Path(jsonl_path)
        self.tokenizer = CharTokenizer(max_text_len)
        self.rows: list[dict[str, str]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"No rows found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image"]).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).unsqueeze(0)

        token_ids, mask = self.tokenizer.encode(row["text"])
        return image_tensor, token_ids, mask
