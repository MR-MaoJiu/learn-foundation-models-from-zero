from __future__ import annotations

"""视觉数据读取。

下载脚本会把 MNIST 保存成：

data/vision/mnist/train/
    labels.csv
    images/000000.png
    images/000001.png

labels.csv 每一行记录：
    image,label

训练时我们读取 PNG 图片，转成 PyTorch Tensor。
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch


class ImageClassificationDataset:
    """简单图像分类数据集。"""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.root = self.csv_path.parent
        self.rows: list[tuple[Path, int]] = []

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self.root / row["image"]
                label = int(row["label"])
                self.rows.append((image_path, label))

        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.rows[index]

        # MNIST 是灰度图，convert("L") 确保只有 1 个通道。
        image = Image.open(image_path).convert("L")

        # 转成 numpy，范围从 0..255 变成 0..1。
        array = np.asarray(image, dtype=np.float32) / 255.0

        # [H, W] -> [C=1, H, W]
        tensor = torch.from_numpy(array).unsqueeze(0)
        target = torch.tensor(label, dtype=torch.long)
        return tensor, target
