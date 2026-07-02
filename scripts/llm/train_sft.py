from __future__ import annotations

"""
监督微调入口脚本。

预训练让模型学会“续写文本”。
SFT 让模型学会“看到用户问题后，输出助手回答”。
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.config import load_json_config, model_config_from_dict
from foundation_models.llm.model import GPT
from foundation_models.llm.sft_data import ChatSFTDataset, read_sft_jsonl, split_examples
from foundation_models.llm.tokenizer import LLMTokenizer
from foundation_models.llm.utils import choose_device, choose_dtype, cosine_lr, save_checkpoint, set_seed


def estimate_loss(model: GPT, dataset: ChatSFTDataset, batch_size: int, eval_iters: int, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = dataset.get_batch(batch_size, device)
            _, loss = model(x, y)
            if loss is not None:
                losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tune the educational LLM.")
    parser.add_argument("--config", required=True, help="Path to SFT config JSON.")
    parser.add_argument(
        "--base-checkpoint",
        default="",
        help="Optional checkpoint to initialize from. Overrides config training.base_checkpoint.",
    )
    args = parser.parse_args()

    raw_config = load_json_config(args.config)
    train_cfg = raw_config["training"]
    set_seed(train_cfg["seed"])

    device = choose_device(train_cfg["device"])
    dtype = choose_dtype(train_cfg["dtype"], device)

    tokenizer = LLMTokenizer(train_cfg["tokenizer_path"])
    raw_config["model"]["vocab_size"] = tokenizer.vocab_size

    model_config = model_config_from_dict(raw_config["model"])
    model = GPT(model_config).to(device)

    base_checkpoint = args.base_checkpoint or train_cfg.get("base_checkpoint", "")
    if base_checkpoint:
        checkpoint = torch.load(base_checkpoint, map_location=device)
        base_model_config = checkpoint["config"]["model"]
        checkpoint_vocab_size = checkpoint["model"]["token_embedding.weight"].shape[0]
        if checkpoint_vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "Base checkpoint vocab size does not match the tokenizer. "
                f"checkpoint vocab={checkpoint_vocab_size}, tokenizer vocab={tokenizer.vocab_size}. "
                "Use the same tokenizer for tokenizer training, pretraining, SFT, and deployment. "
                "You can rerun the full workflow, or pass a checkpoint trained with the current tokenizer."
            )
        base_model_config["vocab_size"] = tokenizer.vocab_size
        model_config = model_config_from_dict(base_model_config)
        raw_config["model"] = base_model_config
        model = GPT(model_config).to(device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded base checkpoint: {base_checkpoint}")
    else:
        print("No base checkpoint provided; SFT starts from random initialization.")

    examples = read_sft_jsonl(train_cfg["sft_jsonl"])
    train_examples, val_examples = split_examples(examples, train_cfg["val_ratio"], train_cfg["seed"])
    train_data = ChatSFTDataset(train_examples, tokenizer, model_config.max_seq_len)
    val_data = ChatSFTDataset(val_examples, tokenizer, model_config.max_seq_len)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_cfg["weight_decay"],
    )

    micro_batch_size = train_cfg["micro_batch_size"]
    batch_size = train_cfg["batch_size"]
    grad_accum_steps = max(1, batch_size // micro_batch_size)

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"SFT train examples: {len(train_data)}")
    print(f"SFT val examples: {len(val_data)}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    model.train()
    start_time = time.time()

    for step in range(train_cfg["max_steps"]):
        lr = cosine_lr(
            step=step,
            max_steps=train_cfg["max_steps"],
            warmup_steps=train_cfg["warmup_steps"],
            learning_rate=train_cfg["learning_rate"],
            min_learning_rate=train_cfg["min_learning_rate"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(grad_accum_steps):
            x, y = train_data.get_batch(micro_batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("SFT loss is None")
                loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        if train_cfg["grad_clip"] > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % train_cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            print(f"step {step:6d} | sft train loss {total_loss:.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s")

        if step > 0 and step % train_cfg["eval_interval"] == 0:
            val_loss = estimate_loss(
                model=model,
                dataset=val_data,
                batch_size=micro_batch_size,
                eval_iters=train_cfg["eval_iters"],
                device=device,
            )
            print(f"step {step:6d} | sft val loss {val_loss:.4f}")

        if step > 0 and step % train_cfg["save_interval"] == 0:
            save_checkpoint(output_dir / f"step_{step}.pt", model, optimizer, step, raw_config)
            save_checkpoint(output_dir / "last.pt", model, optimizer, step, raw_config)
            print(f"SFT checkpoint saved at step {step}")

    save_checkpoint(output_dir / "last.pt", model, optimizer, train_cfg["max_steps"], raw_config)
    print(f"SFT done. Last checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
