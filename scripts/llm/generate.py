from __future__ import annotations

"""
第四步脚本：用训练好的 checkpoint 生成文本。

训练阶段做的是：

    输入 x -> 模型预测 -> 计算 loss -> 更新参数

生成阶段做的是：

    prompt 文本
    -> tokenizer 编码成 token id
    -> 模型预测下一个 token
    -> 把新 token 接到后面
    -> 继续预测下一个 token
    -> 最后 decode 回中文

注意：
模型不是一次性生成一整段话，而是“一次生成一个 token”。
这就是自回归生成。
"""

import argparse
from pathlib import Path
import sys

import torch


# 找到项目根目录，并把 src 加入 Python 导入路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.config import model_config_from_dict
from foundation_models.llm.model import GPT
from foundation_models.llm.tokenizer import LLMTokenizer
from foundation_models.llm.utils import choose_device


def main() -> None:
    """命令行入口。

    示例：

        python scripts/llm/generate.py --checkpoint checkpoints/llm/debug/last.pt --prompt "语言模型的目标是"
    """

    # 有些终端默认编码不一定是 UTF-8。
    # 模型早期生成的字符可能比较乱，如果终端编码不支持就会报错。
    # 这里把输出改成 UTF-8，并用 replace 替换无法显示的字符。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")

    # checkpoint 是训练保存下来的 .pt 文件。
    # 它里面包含模型参数和配置。
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint.",
    )

    # prompt 是你给模型的开头文本。
    # 模型会接着它往后写。
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text.",
    )

    # 最多生成多少个新 token。
    # 注意 token 不等于汉字，一个汉字可能是一个 token，也可能被拆成多个 token。
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )

    # temperature 控制随机性。
    # 低：更保守。
    # 高：更随机。
    # 0 或更低：直接选最高分 token。
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    # top_k 表示每次只从分数最高的 k 个 token 中选择。
    # 它可以减少非常离谱的随机输出。
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    # device="auto" 时，有 GPU 就用 GPU，否则用 CPU。
    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    # 选择 CPU 或 GPU。
    device = choose_device(args.device)

    # 加载 checkpoint。
    #
    # map_location=device 的意思是：
    # checkpoint 里的张量加载到当前选择的设备上。
    # 例如你用 CPU 生成，就加载到 CPU。
    checkpoint = torch.load(args.checkpoint, map_location=device)
    raw_config = checkpoint["config"]

    # 加载 tokenizer。
    # 必须使用训练时同一个 tokenizer，否则 token id 对不上。
    tokenizer = LLMTokenizer(raw_config["training"]["tokenizer_path"])

    # 根据 checkpoint 里的模型配置，重新创建同样结构的模型。
    model_config = model_config_from_dict(raw_config["model"])
    model = GPT(model_config).to(device)

    # 把训练好的参数填进模型。
    model.load_state_dict(checkpoint["model"])

    # eval 模式会关闭 dropout。
    # 生成文本时不希望 dropout 引入额外随机性。
    model.eval()

    # prompt 文本 -> token id。
    # add_bos=True 会在开头加 <bos>，表示序列开始。
    input_ids = tokenizer.encode(args.prompt, add_bos=True)

    # 转成 PyTorch Tensor。
    #
    # tokenizer.encode 返回的是一维列表：
    #   [time]
    #
    # 模型需要 batch 维度，所以外面套一层：
    #   [[time]] -> [batch=1, time]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 生成时不训练，所以不需要梯度。
    # no_grad 可以减少内存占用，也更快。
    with torch.no_grad():
        output = model.generate(
            input_ids=x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # output 形状是 [1, total_time]。
    # output[0] 取出第一个样本。
    # tolist() 把 Tensor 变回 Python list。
    # tokenizer.decode 把 token id 转回字符串。
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
