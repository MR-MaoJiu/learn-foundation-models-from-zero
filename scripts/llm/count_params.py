from __future__ import annotations

"""
辅助脚本：统计模型参数量。

在真正训练前，先看模型有多少参数很重要。

参数量越大：
- 模型容量通常越强。
- 训练越慢。
- 占用显存和内存越多。

这个脚本不会训练模型，只会：

1. 读取 config JSON。
2. 根据 config 创建一个模型。
3. 统计模型里可训练参数的数量。
4. 粗略估算参数文件占用空间。
"""

import argparse
from pathlib import Path
import sys


# 让脚本能 import src/foundation_models/llm。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.config import load_json_config, model_config_from_dict
from foundation_models.llm.model import GPT


def main() -> None:
    """命令行入口。

    示例：

        python scripts/llm/count_params.py --config configs/llm/gpt_50m_8gb.json
    """

    parser = argparse.ArgumentParser(description="Count model parameters.")

    # config JSON 里有 model 配置，例如层数、隐藏维度、注意力头数。
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config JSON.",
    )

    args = parser.parse_args()

    # 读取配置文件。
    raw = load_json_config(args.config)

    # raw["model"] 是一个字典。
    # model_config_from_dict 会把它变成 ModelConfig，并检查字段名是否正确。
    model_config = model_config_from_dict(raw["model"])

    # 创建模型。
    # 注意：这里只创建模型，不加载数据，也不训练。
    model = GPT(model_config)

    # 统计可训练参数。
    total = model.num_parameters()

    print(f"Parameters: {total:,}")

    # 粗略估算只保存“参数本体”需要多少空间。
    #
    # fp32：每个参数 4 字节。
    # fp16/bf16：每个参数 2 字节。
    #
    # 注意：训练时实际占用会更多，因为还要存梯度、优化器状态和中间激活。
    print(f"Approx size in fp32: {total * 4 / 1024**3:.2f} GB")
    print(f"Approx size in bf16/fp16: {total * 2 / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
