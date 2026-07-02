from __future__ import annotations

"""
给一张图片，从候选文本中找最匹配的描述。

这个脚本用于体验 TinyCLIP 学到的图文对齐能力：
1. 读取一张图片。
2. 读取多条候选文本。
3. 分别编码成图片向量和文本向量。
4. 计算相似度。
5. 按相似度从高到低打印文本。
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.multimodal.config import multimodal_config_from_dict
from foundation_models.multimodal.data import CharTokenizer
from foundation_models.multimodal.model import TinyCLIP
from foundation_models.multimodal.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate texts for one image.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained TinyCLIP checkpoint.")
    parser.add_argument("--image", required=True, help="Path to one image.")
    parser.add_argument("--texts", nargs="+", required=True, help="Candidate texts to rank.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = multimodal_config_from_dict(checkpoint["config"]["model"])

    model = TinyCLIP(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 图片预处理必须和训练时保持一致。
    image = Image.open(args.image).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

    # 把每条候选文本编码成 token ids 和 mask。
    tokenizer = CharTokenizer(config.max_text_len)
    encoded = [tokenizer.encode(text) for text in args.texts]
    token_ids = torch.stack([x[0] for x in encoded]).to(device)
    masks = torch.stack([x[1] for x in encoded]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(token_ids, masks)

        # image_features shape: [1, embed_dim]
        # text_features shape: [num_texts, embed_dim]
        # 相乘后得到 [1, num_texts]，表示图片和每条文本的相似度。
        scores = (image_features @ text_features.t())[0]

    order = torch.argsort(scores, descending=True).tolist()
    for idx in order:
        print(f"{scores[idx].item():.4f}  {args.texts[idx]}")


if __name__ == "__main__":
    main()
