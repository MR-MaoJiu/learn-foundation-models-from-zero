from __future__ import annotations

"""
统计 TinyCLIP 多模态模型参数量。

TinyCLIP 由两个编码器组成：
- 图片编码器 ImageEncoder。
- 文本编码器 TextEncoder。

这个脚本只创建模型并统计参数，不读取图片，也不训练。
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.multimodal.config import load_json_config, multimodal_config_from_dict
from foundation_models.multimodal.model import TinyCLIP


def main() -> None:
    parser = argparse.ArgumentParser(description="Count multimodal model parameters.")
    parser.add_argument("--config", required=True, help="Path to a multimodal config JSON file.")
    args = parser.parse_args()

    raw = load_json_config(args.config)
    config = multimodal_config_from_dict(raw["model"])
    model = TinyCLIP(config)
    total = model.num_parameters()

    print(f"Parameters: {total:,}")
    print(f"Approx size in fp32: {total * 4 / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
