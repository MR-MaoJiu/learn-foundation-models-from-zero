from __future__ import annotations

"""
训练入口脚本。

你可以把训练理解成一个重复很多次的循环：

    取一批文本 token
    -> 模型预测下一个 token
    -> 计算预测错得有多离谱，也就是 loss
    -> 反向传播计算每个参数该怎么改
    -> optimizer 更新参数
    -> 重复

这份脚本的任务不是定义模型结构，模型结构在 `src/foundation_models/llm/model.py`。
这里主要负责“把数据、模型、优化器组织起来，并不断训练”。
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import torch

# 让 Python 能找到 src/foundation_models/llm 包。
# 因为我们是用 `python scripts/llm/train.py` 启动脚本，
# 默认导入路径不一定包含项目根目录下的 src。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.config import load_json_config, model_config_from_dict
from foundation_models.llm.data import BinaryTokenDataset
from foundation_models.llm.model import GPT
from foundation_models.llm.tokenizer import LLMTokenizer
from foundation_models.llm.utils import choose_device, choose_dtype, cosine_lr, save_checkpoint, set_seed


def estimate_loss(model: GPT, dataset: BinaryTokenDataset, batch_size: int, eval_iters: int, device: torch.device) -> float:
    """在验证集上估计 loss。

    训练集 loss 下降只能说明模型越来越会拟合训练数据。
    验证集 loss 能告诉我们：模型对没参与训练的数据表现如何。

    eval_iters 表示取多少个 batch 来求平均。
    取太少会抖动，取太多会慢。
    """

    # eval 模式会关闭 dropout 等训练专用行为。
    model.eval()

    losses = []

    # no_grad 表示这里不需要计算梯度。
    # 验证只是看表现，不更新模型参数。
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = dataset.get_batch(batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())

    # 验证完要切回 train 模式，后面还要继续训练。
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    # argparse 用来读取命令行参数。
    # 例如：
    #   python scripts/llm/train.py --config configs/llm/debug.json
    parser = argparse.ArgumentParser(description="Train the educational LLM.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    args = parser.parse_args()

    # 读取 JSON 配置。
    # raw_config 是一个普通 dict，里面有 model 和 training 两块。
    config_path = Path(args.config)
    raw_config = load_json_config(config_path)
    train_cfg = raw_config["training"]

    # 固定随机种子。
    # 这不能保证每次完全一样，但能减少随机性，方便学习和调试。
    set_seed(train_cfg["seed"])

    # device 决定用 CPU 还是 GPU。
    # 配置里是 "auto" 时，有 CUDA 就用显卡，否则用 CPU。
    device = choose_device(train_cfg["device"])

    # dtype 决定用什么数字精度。
    # GPU 上通常用 fp16/bf16 更省显存；CPU 上用 fp32 更稳。
    dtype = choose_dtype(train_cfg["dtype"], device)

    # 加载 tokenizer。
    # tokenizer 负责 text <-> token id。
    tokenizer = LLMTokenizer(train_cfg["tokenizer_path"])

    # 用 tokenizer 的真实词表大小覆盖配置。
    # 这样可以避免模型 vocab_size 和 tokenizer 文件不一致。
    raw_config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = model_config_from_dict(raw_config["model"])

    # 加载训练集和验证集。
    # BinaryTokenDataset 会从 .bin 文件里随机切片，构造 x/y。
    train_data = BinaryTokenDataset(train_cfg["train_bin"], model_config.max_seq_len, model_config.vocab_size)
    val_data = BinaryTokenDataset(train_cfg["val_bin"], model_config.max_seq_len, model_config.vocab_size)

    # 创建模型并移动到 device。
    # 如果 device 是 cuda，模型参数会放进显存。
    model = GPT(model_config).to(device)

    # AdamW 是训练 Transformer 常用的优化器。
    #
    # optimizer 的作用：
    # - 根据每个参数的梯度，决定参数要往哪个方向改。
    # - learning_rate 控制每一步改多大。
    # - weight_decay 让参数不要无限变大，有一点正则化作用。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_cfg["weight_decay"],
    )

    # batch_size 是“理论上的总 batch”。
    # micro_batch_size 是“一次真的放进显存的 batch”。
    #
    # 如果显存不够一次放 batch_size，就分几次放：
    #   batch_size = 64
    #   micro_batch_size = 4
    #   grad_accum_steps = 16
    #
    # 意思是累积 16 次小 batch 的梯度，再更新一次参数。
    micro_batch_size = train_cfg["micro_batch_size"]
    batch_size = train_cfg["batch_size"]
    grad_accum_steps = max(1, batch_size // micro_batch_size)

    # AMP = Automatic Mixed Precision，自动混合精度。
    # 在支持的 GPU 上，它能减少显存占用、加快训练。
    # CPU 上不启用。
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if use_amp
        else nullcontext()
    )

    # GradScaler 只在 fp16 训练时需要。
    # 它能减少 fp16 因数值太小导致的梯度下溢问题。
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    # 创建 checkpoint 输出目录。
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    start_time = time.time()

    # train 模式会启用 dropout。
    model.train()

    # 主训练循环。
    # step 表示第几次参数更新。
    for step in range(train_cfg["max_steps"]):
        # 计算当前学习率。
        # 一开始 warmup：学习率从小慢慢升高。
        # 后面 cosine decay：学习率慢慢下降。
        lr = cosine_lr(
            step=step,
            max_steps=train_cfg["max_steps"],
            warmup_steps=train_cfg["warmup_steps"],
            learning_rate=train_cfg["learning_rate"],
            min_learning_rate=train_cfg["min_learning_rate"],
        )

        # 把学习率写进 optimizer。
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 清空上一轮的梯度。
        # PyTorch 默认会累加梯度，所以每次参数更新前要清零。
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        # 梯度累积循环。
        # 每次处理一个 micro batch，多次 backward 后再 optimizer.step。
        for _ in range(grad_accum_steps):
            # x 是输入，y 是答案。
            #
            # 如果一段 token 是：
            #   [10, 20, 30, 40, 50]
            #
            # max_seq_len=4 时：
            #   x = [10, 20, 30, 40]
            #   y = [20, 30, 40, 50]
            #
            # 也就是每个位置预测下一个 token。
            x, y = train_data.get_batch(micro_batch_size, device)

            # autocast_ctx 在 GPU 混合精度时会自动把部分计算改成 fp16/bf16。
            with autocast_ctx:
                _, loss = model(x, y)

                # 因为我们会 backward 多次再更新一次参数，
                # 所以每个小 batch 的 loss 要除以累积次数。
                # 这样总梯度大小才接近一次大 batch。
                loss = loss / grad_accum_steps

            # backward：反向传播。
            # 它会根据 loss，计算每个参数的梯度。
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # 梯度裁剪。
        # 如果梯度太大，训练会不稳定。
        # clip_grad_norm_ 会把整体梯度限制在 grad_clip 以内。
        if train_cfg["grad_clip"] > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])

        # optimizer.step：真正更新模型参数。
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # 打印训练日志。
        # train loss 下降说明模型在训练集上学到了东西。
        if step % train_cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            print(f"step {step:6d} | train loss {total_loss:.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s")

        # 定期在验证集上评估。
        # 如果 train loss 降、val loss 升，可能是过拟合。
        if step > 0 and step % train_cfg["eval_interval"] == 0:
            val_loss = estimate_loss(
                model=model,
                dataset=val_data,
                batch_size=micro_batch_size,
                eval_iters=train_cfg["eval_iters"],
                device=device,
            )
            print(f"step {step:6d} | val loss {val_loss:.4f}")

        # 定期保存 checkpoint。
        # checkpoint 里包含模型参数、优化器状态、step 和配置。
        if step > 0 and step % train_cfg["save_interval"] == 0:
            save_checkpoint(output_dir / f"step_{step}.pt", model, optimizer, step, raw_config)
            save_checkpoint(output_dir / "last.pt", model, optimizer, step, raw_config)
            print(f"checkpoint saved at step {step}")

    # 训练结束时一定保存最后一次模型。
    save_checkpoint(output_dir / "last.pt", model, optimizer, train_cfg["max_steps"], raw_config)
    print(f"Training done. Last checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
