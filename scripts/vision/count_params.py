from __future__ import annotations

"""
统计视觉 CNN 参数量。

这个脚本不会下载数据，也不会训练模型。
它只做三件事：
1. 读取配置文件。
2. 按配置创建 SmallCNN。
3. 统计模型里需要训练的参数数量。

学习阶段建议先跑这个脚本，确认模型规模不会太大。
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.vision.config import load_json_config, vision_config_from_dict
from foundation_models.vision.model import SmallCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Count vision model parameters.")
    parser.add_argument("--config", required=True, help="Path to a vision config JSON file.")
    args = parser.parse_args()

    raw = load_json_config(args.config)
    config = vision_config_from_dict(raw["model"])
    model = SmallCNN(config)
    total = model.num_parameters()

    print(f"Parameters: {total:,}")

    # fp32 表示每个参数用 4 字节保存。
    # 这只是参数本身大小，训练时还会有梯度、优化器状态和中间激活，占用会更多。
    print(f"Approx size in fp32: {total * 4 / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
