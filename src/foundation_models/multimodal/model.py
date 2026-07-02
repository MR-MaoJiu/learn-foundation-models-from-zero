from __future__ import annotations

"""
一个极小版 CLIP 风格图文对齐模型。

CLIP 类模型的核心想法：
1. 图片编码器把图片变成向量。
2. 文本编码器把文本变成向量。
3. 匹配的图片向量和文本向量应该更接近。
4. 不匹配的图片向量和文本向量应该更远。

这个 TinyCLIP 非常小，只用于学习原理：
- 图像编码器：小 CNN。
- 文本编码器：字符 embedding + 平均池化。
- 训练目标：对比学习 loss。
"""

import torch
from torch import nn
import torch.nn.functional as F

from .config import MultimodalConfig


class ImageEncoder(nn.Module):
    """把图片编码成一个向量。

    输入：
        images shape = [batch, image_channels, image_size, image_size]

    对 MNIST 来说：
        images shape = [batch, 1, 28, 28]

    输出：
        image_features shape = [batch, embed_dim]
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        c = config.hidden_channels

        # nn.Sequential 会按照顺序执行里面的层。
        # 这里先用两层卷积和池化提取图片特征，再拉平成一维向量，最后映射到 embed_dim。
        self.net = nn.Sequential(
            nn.Conv2d(config.image_channels, c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(c * 2 * 7 * 7, config.embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """执行图片编码。"""

        return self.net(images)


class TextEncoder(nn.Module):
    """把短文本编码成一个向量。

    这里使用字符级编码，是为了足够透明：
    - 每个字符先变成 token id。
    - token id 通过 embedding 表查到一个向量。
    - 对所有真实字符的向量求平均。
    - 用线性层投影到和图片向量一样的维度。
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()

        # padding_idx=0 表示 id 为 0 的位置是补齐符号 padding。
        # PyTorch 会让这个位置的 embedding 不参与正常学习。
        self.embedding = nn.Embedding(config.vocab_size, config.text_embed_dim, padding_idx=0)
        self.proj = nn.Linear(config.text_embed_dim, config.embed_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """执行文本编码。

        参数：
            token_ids shape = [batch, text_len]
            attention_mask shape = [batch, text_len]

        attention_mask 中：
            1 表示真实字符。
            0 表示 padding，不应该参与平均。
        """

        # x shape: [batch, text_len, text_embed_dim]
        x = self.embedding(token_ids)

        # mask shape: [batch, text_len, 1]
        # unsqueeze(-1) 是为了让 mask 可以和 x 的最后一个维度对齐相乘。
        mask = attention_mask.unsqueeze(-1).float()
        x = x * mask

        # 对真实字符求平均。
        # clamp_min(1.0) 是为了防止极端情况下除以 0。
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = x.sum(dim=1) / denom

        return self.proj(pooled)


class TinyCLIP(nn.Module):
    """极小图文对齐模型。"""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)

        # logit_scale 是可学习温度参数。
        # 它控制相似度矩阵的“尖锐程度”。
        # 初学时先记住：它会影响 softmax 区分正确配对和错误配对的力度。
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """把图片编码并归一化。

        F.normalize 会让向量长度接近 1。
        这样后面的点积主要反映方向相似度，而不是向量长度。
        """

        features = self.image_encoder(images)
        return F.normalize(features, dim=-1)

    def encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """把文本编码并归一化。"""

        features = self.text_encoder(token_ids, attention_mask)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回相似度矩阵和对比学习 loss。

        如果 batch_size = 4，相似度矩阵 logits 的形状是 [4, 4]：

                     text_0  text_1  text_2  text_3
            image_0    ?       ?       ?       ?
            image_1    ?       ?       ?       ?
            image_2    ?       ?       ?       ?
            image_3    ?       ?       ?       ?

        对角线位置是正确图文配对：
            image_0 对 text_0
            image_1 对 text_1
            image_2 对 text_2
            image_3 对 text_3
        """

        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids, attention_mask)

        # image_features @ text_features.t() 是矩阵乘法。
        # 结果中第 i 行第 j 列表示第 i 张图片和第 j 条文本的相似度。
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * image_features @ text_features.t()

        # labels = [0, 1, 2, ..., batch_size-1]
        # 对第 i 张图片来说，正确文本就是第 i 条文本。
        labels = torch.arange(images.size(0), device=images.device)

        # 图片找文本：每一行做一次分类，正确答案是对角线。
        image_to_text_loss = F.cross_entropy(logits, labels)

        # 文本找图片：转置后每一行代表一条文本对所有图片的分数。
        text_to_image_loss = F.cross_entropy(logits.t(), labels)

        loss = (image_to_text_loss + text_to_image_loss) / 2
        return logits, loss

    def num_parameters(self) -> int:
        """统计可训练参数数量。"""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
