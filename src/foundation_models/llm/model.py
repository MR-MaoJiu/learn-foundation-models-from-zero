from __future__ import annotations

"""
这个文件是整个项目最核心的地方：从 0 实现一个 decoder-only Transformer。

你可以先记住一条主线：

    token id -> 词向量 -> 多层 Transformer -> 词表分数 -> loss

更具体一点：

1. `input_ids` 是一批 token 编号，例如：
       [[12, 45, 88, 3],
        [90, 17, 26, 4]]

2. `Embedding` 把每个编号变成一条向量。
   如果隐藏维度 n_embd=512，那么一个 token 会变成 512 个数字。

3. Transformer Block 重复很多次。
   每个 Block 里有两件事：
   - Attention：让当前位置从前文里找信息。
   - MLP：对每个位置的向量做非线性加工。

4. `lm_head` 把隐藏向量变成“词表中每个 token 的分数”。
   如果词表大小是 8000，那么每个位置会输出 8000 个分数。

5. 训练时用 labels 计算交叉熵 loss。
   loss 越低，说明模型越会预测下一个 token。
"""

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm：让每个 token 的隐藏向量数值更稳定。

    为什么需要归一化？
    - 神经网络层数多了以后，数值可能越来越大或越来越小。
    - 数值不稳定会让训练变难，loss 可能抖动甚至爆炸。
    - Norm 层的作用就是把向量拉回一个比较稳定的尺度。

    RMSNorm 和 LayerNorm 很像，但它不减去均值，只按均方根缩放。
    许多 LLaMA 风格模型都使用 RMSNorm。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        # eps 是一个很小的数，防止除以 0。
        self.eps = eps

        # weight 是可训练参数，形状是 [hidden]。
        # 归一化后，模型还可以自己学习每个维度应该放大或缩小多少。
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状是 [batch, time, hidden]
        #
        # batch：一次训练多少段文本
        # time：每段文本有多少 token
        # hidden：每个 token 用多少维向量表示

        # x.pow(2)：每个数字平方。
        # mean(dim=-1)：在 hidden 维度求平均。
        # keepdim=True：保留维度，方便后面和 x 相乘。
        #
        # 结果形状是 [batch, time, 1]。
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # torch.rsqrt(a) 等价于 1 / sqrt(a)。
        # x 乘上 rms_scale 后，每个 token 向量的整体尺度会变稳定。
        rms_scale = torch.rsqrt(mean_square + self.eps)

        return self.weight * x * rms_scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 的辅助函数：把成对维度旋转 90 度。

    假设某一对维度是 [a, b]，这个函数会变成 [-b, a]。

    RoPE 的核心思想：
    - 位置不是直接加到 embedding 上。
    - 而是通过 sin/cos 旋转 query 和 key。
    - 旋转后，注意力分数会自然带上相对位置信息。
    """

    # 取偶数位置维度：0, 2, 4, ...
    x1 = x[..., ::2]

    # 取奇数位置维度：1, 3, 5, ...
    x2 = x[..., 1::2]

    # stack 后最后一维变成 [-x2, x1]，再 flatten 回原来的 head_dim。
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预先计算 RoPE 需要的 cos 和 sin。

    为什么叫 cache？
    - 同一个 forward 里，每一层 attention 都会用同样的位置编码。
    - 提前算好 cos/sin，后面每层复用，代码更清晰。

    返回：
    - cos 形状：[1, 1, time, head_dim]
    - sin 形状：[1, 1, time, head_dim]

    前两个 1 是为了自动广播到 batch 和 heads。
    """

    # positions = [0, 1, 2, ..., seq_len-1]
    # 表示每个 token 的位置。
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)

    # dims = [0, 2, 4, ...]
    # RoPE 每两个维度一组，所以这里步长是 2。
    dims = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)

    # 不同维度使用不同频率。
    # 低维变化慢，高维变化快，让模型能同时感知近距离和远距离位置。
    inv_freq = 1.0 / (theta ** (dims / head_dim))

    # outer 得到形状 [time, head_dim/2]。
    # 每一行是某个位置，每一列是某个旋转频率。
    freqs = torch.outer(positions, inv_freq)

    # 每个频率复制两次，让它能对应完整 head_dim。
    # 例如 [f0, f1] -> [f0, f0, f1, f1]
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

    # 增加两个维度后，cos/sin 可以和 q/k 的形状自动对齐：
    # q/k 形状：[batch, heads, time, head_dim]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """把 RoPE 应用到 query 或 key。

    输入 x 的形状：[batch, heads, time, head_dim]

    公式可以粗略理解成：
        新向量 = 原向量 * cos + 旋转后的原向量 * sin

    这样做不会改变向量形状，只是让向量携带位置信息。
    """

    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """因果自注意力层。

    Attention 解决的问题：
    - 当前 token 预测下一个 token 时，应该参考前文哪些 token？
    - 比如“我今天很饿，所以想吃”，模型应该关注“饿”来预测“饭”等词。

    为什么叫 causal？
    - 语言模型生成时只能从左往右写。
    - 当前位置不能看未来 token。
    - 训练时也必须遵守这个规则，否则模型会偷看答案。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        # q/k/v 是 attention 的三个核心向量：
        #
        # q = query：当前位置想找什么信息？
        # k = key：每个历史位置提供什么索引？
        # v = value：每个历史位置真正携带的信息内容。
        #
        # 这三个 Linear 都是可训练层。
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # 多个头的结果拼回 hidden 后，再经过一个输出投影。
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状：[batch, time, hidden]
        batch, time, hidden = x.shape

        # 第一步：从同一个 x 里投影出 q、k、v。
        # 形状都还是 [batch, time, hidden]。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 第二步：把 hidden 拆成多个注意力头。
        #
        # 原来 hidden = n_head * head_dim。
        # view 后：[batch, time, heads, head_dim]
        # transpose 后：[batch, heads, time, head_dim]
        #
        # 为什么要多头？
        # - 一个头可以关注语法关系。
        # - 另一个头可以关注主题词。
        # - 另一个头可以关注标点或句子边界。
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

        # 第三步：给 q 和 k 加 RoPE 位置编码。
        # 注意：RoPE 通常只加到 q/k，不加到 v。
        q = apply_rope(q, cos[:, :, :time, :], sin[:, :, :time, :])
        k = apply_rope(k, cos[:, :, :time, :], sin[:, :, :time, :])

        # 第四步：真正计算注意力。
        #
        # PyTorch 的 scaled_dot_product_attention 做了这些事：
        # 1. q 和 k 做点积，得到“当前位置对历史位置的关注分数”。
        # 2. 除以 sqrt(head_dim)，避免分数太大。
        # 3. 因为 is_causal=True，会屏蔽未来位置。
        # 4. softmax，把分数变成概率。
        # 5. 用概率加权求和 v。
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # 第五步：把多个头拼回一个 hidden 向量。
        # [batch, heads, time, head_dim]
        # -> [batch, time, heads, head_dim]
        # -> [batch, time, hidden]
        attn = attn.transpose(1, 2).contiguous().view(batch, time, hidden)

        # 第六步：输出投影，并做 dropout。
        return self.resid_dropout(self.o_proj(attn))


class SwiGLU(nn.Module):
    """Transformer 里的前馈网络 MLP。

    Attention 负责“不同 token 之间互相看”。
    MLP 负责“每个 token 自己内部做思考和变换”。

    SwiGLU 是一种带门控的 MLP：
    - gate_proj 决定哪些信息应该通过。
    - up_proj 提供候选信息。
    - down_proj 把大维度再压回 hidden 维度。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # 从 hidden 扩展到 intermediate_size。
        # 例如 512 -> 2048。
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)

        # 再从 intermediate_size 压回 hidden。
        # 例如 2048 -> 512。
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu 是一种激活函数，让网络能表达非线性关系。
        #
        # gate * up 的意思像一个开关：
        # - gate 大，信息容易通过。
        # - gate 小，信息被压住。
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    """一个 Transformer Block。

    顺序是：
        输入 x
        -> RMSNorm
        -> Attention
        -> 残差连接
        -> RMSNorm
        -> MLP
        -> 残差连接

    残差连接 `x = x + new_info` 很重要：
    - 它让模型保留原来的信息。
    - 它让深层网络更容易训练。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # 先归一化，再 attention，然后把结果加回 x。
        x = x + self.attn(self.attn_norm(x), cos, sin)

        # 再归一化，再 MLP，然后把结果加回 x。
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """decoder-only Transformer 语言模型。

    decoder-only 的意思是：
    - 它只做从左到右的文本预测。
    - GPT、LLaMA、Qwen、Mistral 这一类大语言模型基本都是这种路线。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # token_embedding 是一张表。
        # 表的行数 = vocab_size，表示词表里有多少 token。
        # 表的列数 = n_embd，表示每个 token 用多少维向量表示。
        #
        # 输入 token id 后，Embedding 会查表取出对应向量。
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # dropout 会随机把一部分数值置 0，用来降低过拟合。
        self.dropout = nn.Dropout(config.dropout)

        # 堆叠多个 TransformerBlock。
        # n_layer 越大，模型越深，能力更强，但更慢、更吃显存。
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # 最后一层归一化。
        self.norm = RMSNorm(config.n_embd)

        # lm_head 把 hidden 向量映射回词表大小。
        # 例如 [batch, time, 512] -> [batch, time, 8000]
        #
        # 输出的 8000 个数字不是概率，而是 logits，后面会经过 softmax。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：
        # 输入 embedding 表和输出 lm_head 表共享同一份参数。
        #
        # 直觉：
        # - 输入时 token 需要一个向量表示。
        # - 输出时也要判断哪个 token 最匹配当前向量。
        # - 共享权重可以减少参数，也常常让语言模型效果更好。
        self.lm_head.weight = self.token_embedding.weight

        # 初始化所有 Linear 和 Embedding 的权重。
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """初始化模型参数。

        模型一开始什么都不会，所以参数从随机数开始。
        训练的过程就是不断修改这些随机参数，让 loss 降低。
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """模型前向传播。

        参数：
        - input_ids：输入 token，形状 [batch, time]
        - labels：答案 token，形状 [batch, time]

        返回：
        - logits：每个位置对词表所有 token 的打分
        - loss：如果传了 labels，就返回训练损失；否则返回 None

        一个小例子：
            input_ids = [我, 喜欢, 学习]
            labels    = [喜欢, 学习, <eos>]

        模型在第 1 个位置看到“我”，要预测“喜欢”。
        模型在第 2 个位置看到“我 喜欢”，要预测“学习”。
        模型在第 3 个位置看到“我 喜欢 学习”，要预测结束或下文。
        """

        batch, time = input_ids.shape
        if time > self.config.max_seq_len:
            raise ValueError(f"Sequence length {time} exceeds max_seq_len {self.config.max_seq_len}")

        device = input_ids.device

        # 根据当前序列长度创建 RoPE 缓存。
        cos, sin = build_rope_cache(
            seq_len=time,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            device=device,
        )

        # token id -> token embedding。
        # [batch, time] -> [batch, time, hidden]
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # 依次通过每一层 Transformer。
        for block in self.blocks:
            x = block(x, cos, sin)

        # 最后一层归一化。
        x = self.norm(x)

        # hidden -> vocab logits。
        # [batch, time, hidden] -> [batch, time, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # F.cross_entropy 要求：
            # - 输入 logits 形状是 [样本数, 类别数]
            # - 标签 labels 形状是 [样本数]
            #
            # 现在 logits 是 [batch, time, vocab_size]，
            # 所以 reshape 成 [batch*time, vocab_size]。
            #
            # labels 是 [batch, time]，
            # 所以 reshape 成 [batch*time]。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """根据 prompt 生成新 token。

        生成过程是一个循环：
        1. 把已有 token 喂给模型。
        2. 取最后一个位置的 logits。
        3. 从 logits 里选出下一个 token。
        4. 把新 token 拼到原序列后面。
        5. 重复。

        temperature：
        - 越低，越保守，越容易选最高分 token。
        - 越高，越随机，越容易发散。

        top_k：
        - 只从分数最高的 k 个 token 里采样。
        - 可以避免模型从很离谱的 token 里随机选。
        """

        for _ in range(max_new_tokens):
            # 如果上下文太长，只保留最后 max_seq_len 个 token。
            context = input_ids[:, -self.config.max_seq_len :]

            # logits 形状：[batch, time, vocab_size]
            logits, _ = self(context)

            # 只关心最后一个位置，因为它负责预测下一个 token。
            # 形状：[batch, vocab_size]
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                # temperature <= 0 时完全贪心：选分数最高的 token。
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                # temperature 缩放 logits。
                # 除以小于 1 的数会让最高分更突出。
                # 除以大于 1 的数会让分布更平，随机性更强。
                next_logits = next_logits / temperature

                if top_k > 0:
                    # 找到 top_k 中最低的那个分数作为 cutoff。
                    values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                    cutoff = values[:, [-1]]

                    # 不在 top_k 里的 token 分数改成 -inf。
                    # softmax 后它们的概率会变成 0。
                    next_logits = torch.where(
                        next_logits < cutoff,
                        torch.full_like(next_logits, -math.inf),
                        next_logits,
                    )

                # softmax 把 logits 转成概率。
                probs = F.softmax(next_logits, dim=-1)

                # 按概率随机抽一个 token。
                next_id = torch.multinomial(probs, num_samples=1)

            # 把新 token 拼到序列末尾。
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    def num_parameters(self) -> int:
        """统计可训练参数数量。

        这个数字越大，模型容量通常越强，但训练越慢、越吃显存。
        """

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
