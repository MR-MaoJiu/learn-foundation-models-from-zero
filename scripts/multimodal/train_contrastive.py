from __future__ import annotations

"""
训练 TinyCLIP 图文对齐模型。

和普通分类训练不同，这里不是让模型直接预测 0 到 9。
我们要让模型学会：
- 匹配的图片和文本，向量相似度高。
- 不匹配的图片和文本，向量相似度低。

这类训练叫对比学习 contrastive learning。
"""

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.multimodal.config import load_json_config, multimodal_config_from_dict
from foundation_models.multimodal.data import ImageTextDataset
from foundation_models.multimodal.model import TinyCLIP
from foundation_models.multimodal.utils import choose_device, save_checkpoint, set_seed


def evaluate(model: TinyCLIP, loader: DataLoader, device: torch.device) -> float:
    """在验证集上计算平均对比学习 loss。

    这里不计算准确率，是因为图文对齐任务更常用 recall、ranking 等指标。
    为了保持项目简单，先观察 loss 是否下降。
    """

    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for images, token_ids, masks in loader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            masks = masks.to(device)

            # model 返回两个值：
            # logits: 图文相似度矩阵。
            # loss:  对比学习损失。
            _, loss = model(images, token_ids, masks)
            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)

    model.train()
    return total_loss / total_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyCLIP on MNIST image-text pairs.")
    parser.add_argument("--config", required=True, help="Path to a multimodal config JSON file.")
    args = parser.parse_args()

    raw = load_json_config(args.config)
    train_cfg = raw["training"]
    model_cfg = multimodal_config_from_dict(raw["model"])

    set_seed(train_cfg["seed"])
    device = choose_device(train_cfg["device"])

    # ImageTextDataset 会读取 JSONL。
    # 每条样本返回：图片张量、文本 token ids、文本 mask。
    train_set = ImageTextDataset(train_cfg["train_jsonl"], model_cfg.max_text_len)
    val_set = ImageTextDataset(train_cfg["val_jsonl"], model_cfg.max_text_len)
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False)

    model = TinyCLIP(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")

    step = 0
    model.train()

    for epoch in range(train_cfg["max_epochs"]):
        for images, token_ids, masks in train_loader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            masks = masks.to(device)

            # 前向传播后，TinyCLIP 内部会：
            # 1. 把图片编码成 image_features。
            # 2. 把文本编码成 text_features。
            # 3. 计算 image_features 和 text_features 的相似度矩阵。
            # 4. 用对角线作为正确配对计算 loss。
            _, loss = model(images, token_ids, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % train_cfg["log_interval"] == 0:
                print(f"epoch {epoch} step {step} train loss {loss.item():.4f}")
            step += 1

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val loss {val_loss:.4f}")

    output_dir = Path(train_cfg["output_dir"])
    save_checkpoint(output_dir / "last.pt", model, raw)
    print(f"Saved checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
