from __future__ import annotations

"""
用训练好的视觉模型预测单张 MNIST 图片。

训练脚本保存的 checkpoint 里包含两类信息：
1. model：模型参数，也就是训练后学到的权重。
2. config：创建模型需要的结构配置，例如通道数、类别数。

所以推理时不需要再单独传 config 文件，只要传 checkpoint 和图片路径即可。
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.vision.config import vision_config_from_dict
from foundation_models.vision.model import SmallCNN
from foundation_models.vision.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one MNIST image.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint, such as last.pt.")
    parser.add_argument("--image", required=True, help="Path to one PNG image.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    args = parser.parse_args()

    device = choose_device(args.device)

    # map_location=device 表示即使模型是在 GPU 上保存的，也可以加载到当前设备。
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # checkpoint["config"]["model"] 是训练时保存下来的模型结构配置。
    config = vision_config_from_dict(checkpoint["config"]["model"])

    model = SmallCNN(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 和训练时保持一致：转灰度、转 float32、归一化到 0..1。
    image = Image.open(args.image).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0

    # 原始 array shape 是 [28, 28]。
    # 第一次 unsqueeze 加 channel 维度 -> [1, 28, 28]。
    # 第二次 unsqueeze 加 batch 维度 -> [1, 1, 28, 28]。
    x = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred].item())

    print(f"Predicted digit: {pred}")
    print(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
