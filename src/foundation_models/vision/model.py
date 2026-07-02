from __future__ import annotations

"""一个小型 CNN 图像分类模型。

CNN = Convolutional Neural Network，卷积神经网络。

它适合处理图像，因为卷积层会在图片上滑动小窗口，寻找局部模式：
- 边缘
- 笔画
- 拐角
- 简单形状

MNIST 数字很简单，所以一个小 CNN 就能学得不错。
"""

import torch
from torch import nn
import torch.nn.functional as F

from .config import VisionConfig


class SmallCNN(nn.Module):
    """小型卷积分类器。

    输入：
        x shape = [batch, channels, height, width]

    对 MNIST 来说：
        x shape = [batch, 1, 28, 28]

    输出：
        logits shape = [batch, 10]

    10 个数字分别表示模型认为这张图属于 0 到 9 的分数。
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        c = config.hidden_channels

        # 第一层卷积：
        # 从 1 个灰度通道，提取 c 个特征通道。
        self.conv1 = nn.Conv2d(config.channels, c, kernel_size=3, padding=1)

        # 第二层卷积：
        # 从 c 个特征通道，提取 2c 个更丰富的特征。
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=3, padding=1)

        # 第三层卷积：
        # 继续提取更抽象的图像特征。
        self.conv3 = nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(config.dropout)

        # 两次 max_pool2d 会让 28x28 变成 7x7。
        # 所以最后 flatten 后的维度是 2c * 7 * 7。
        self.fc = nn.Linear(c * 2 * 7 * 7, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        每一步形状大致是：
        [B, 1, 28, 28]
        -> conv1 -> [B, C, 28, 28]
        -> pool  -> [B, C, 14, 14]
        -> conv2 -> [B, 2C, 14, 14]
        -> pool  -> [B, 2C, 7, 7]
        -> conv3 -> [B, 2C, 7, 7]
        -> flatten -> [B, 2C*7*7]
        -> fc -> [B, 10]
        """

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits

    def num_parameters(self) -> int:
        """统计可训练参数数量。"""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
