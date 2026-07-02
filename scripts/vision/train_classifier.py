from __future__ import annotations

"""
训练一个小型 CNN 视觉分类模型。

完整流程：
1. 读取配置文件，知道数据在哪里、模型多大、训练多少轮。
2. 用 Dataset 和 DataLoader 读取图片和标签。
3. 创建 CNN 模型。
4. 对每个 batch 做前向传播，得到 10 个类别分数。
5. 用交叉熵 loss 判断模型错得多严重。
6. 反向传播计算梯度。
7. 优化器根据梯度更新模型参数。
8. 每轮结束后在验证集上计算 loss 和准确率。
"""

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 这个脚本位于 scripts/vision/ 下。
# parents[2] 会回到项目根目录，方便后面导入 src 里的代码。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.vision.config import load_json_config, vision_config_from_dict
from foundation_models.vision.data import ImageClassificationDataset
from foundation_models.vision.model import SmallCNN
from foundation_models.vision.utils import choose_device, save_checkpoint, set_seed


def evaluate(model: SmallCNN, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """在验证集上计算平均 loss 和准确率。

    这个函数和训练循环很像，但有两个关键区别：
    1. 使用 model.eval()，关闭 dropout 等训练专用行为。
    2. 使用 torch.no_grad()，不保存梯度，节省内存和计算。

    返回：
        (平均 loss, 准确率)
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for x, y in loader:
            # x shape: [batch, 1, 28, 28]
            # y shape: [batch]
            x = x.to(device)
            y = y.to(device)

            # logits shape: [batch, 10]
            # 10 个数字分别是模型认为图片属于 0..9 的分数。
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # loss.item() 是当前 batch 的平均 loss。
            # 乘以 batch size 后累加，最后再除以总样本数，得到全验证集平均 loss。
            total_loss += loss.item() * x.size(0)

            # argmax 取分数最高的类别作为预测结果。
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()
            total_items += x.size(0)

    model.train()
    return total_loss / total_items, total_correct / total_items


def main() -> None:
    """训练脚本入口。"""

    parser = argparse.ArgumentParser(description="Train a small CNN on MNIST.")
    parser.add_argument("--config", required=True, help="Path to a vision config JSON file.")
    args = parser.parse_args()

    raw = load_json_config(args.config)
    train_cfg = raw["training"]
    model_cfg = vision_config_from_dict(raw["model"])

    # 固定随机种子后，同样的配置更容易复现相近结果。
    set_seed(train_cfg["seed"])
    device = choose_device(train_cfg["device"])

    # Dataset 负责“按索引读取一条样本”。
    # DataLoader 负责“把多条样本拼成 batch，并处理 shuffle”。
    train_set = ImageClassificationDataset(train_cfg["train_csv"])
    val_set = ImageClassificationDataset(train_cfg["val_csv"])
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False)

    model = SmallCNN(model_cfg).to(device)

    # AdamW 是常用优化器。
    # learning_rate 控制每次参数更新的步子大小。
    # weight_decay 是一种正则化，能稍微抑制参数无限变大。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")

    model.train()
    global_step = 0

    for epoch in range(train_cfg["max_epochs"]):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # 前向传播：模型根据当前参数做一次预测。
            logits = model(x)

            # 交叉熵：分类任务最常用的 loss。
            # 它会惩罚“正确类别分数不够高”的情况。
            loss = F.cross_entropy(logits, y)

            # 清空上一轮留下的梯度。
            # PyTorch 默认会累加梯度，所以每次更新前都要清空。
            optimizer.zero_grad(set_to_none=True)

            # 反向传播：根据 loss 计算每个参数的梯度。
            loss.backward()

            # 参数更新：优化器真正修改模型权重。
            optimizer.step()

            if global_step % train_cfg["log_interval"] == 0:
                print(f"epoch {epoch} step {global_step} train loss {loss.item():.4f}")
            global_step += 1

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val loss {val_loss:.4f} val acc {val_acc:.4f}")

    output_dir = Path(train_cfg["output_dir"])
    save_checkpoint(output_dir / "last.pt", model, raw)
    print(f"Saved checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
