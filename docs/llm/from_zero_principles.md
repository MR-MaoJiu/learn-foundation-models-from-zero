# 从 0 理解这个小型 LLM

这份文档不追求术语完整，而是帮你先把主线抓住。

## 1. LLM 到底在学什么

语言模型最核心的任务只有一句话：

```text
根据前面的 token，预测下一个 token。
```

例如一句话：

```text
我 喜欢 学习
```

训练时会变成：

```text
输入 x: 我 喜欢
答案 y: 喜欢 学习
```

也就是说：

- 看到“我”，模型要预测“喜欢”。
- 看到“我 喜欢”，模型要预测“学习”。

模型不是一开始就懂中文。它只是从大量例子中不断调整参数，让自己预测下一个 token 的错误越来越小。

## 2. 为什么要 tokenizer

神经网络不能直接处理文字，只能处理数字。

所以需要 tokenizer：

```text
语言模型 -> [128, 42, 905]
```

这些数字叫 token id。

训练完模型生成文本时，又会反过来：

```text
[128, 42, 905] -> 语言模型
```

本项目里相关文件：

- `scripts/llm/train_tokenizer.py`：训练 tokenizer。
- `src/foundation_models/llm/tokenizer.py`：封装 encode/decode。

## 3. x 和 y 是怎么来的

假设 token 序列是：

```text
[10, 20, 30, 40, 50]
```

如果上下文长度是 4，那么训练样本是：

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

它们错开一位。

这表示：

- 输入 10，答案是 20。
- 输入 10,20，答案是 30。
- 输入 10,20,30，答案是 40。
- 输入 10,20,30,40，答案是 50。

本项目里相关文件：

- `scripts/llm/prepare_data.py`：把文本变成 `train.bin` / `val.bin`。
- `src/foundation_models/llm/data.py`：从 `.bin` 里随机切出 x/y。

## 4. Embedding 是什么

token id 只是编号，本身没有语义。

Embedding 是一张表：

```text
token id -> 向量
```

例如：

```text
12 -> [0.1, -0.3, 0.8, ...]
```

如果隐藏维度是 512，那么每个 token 会变成 512 个数字。

模型训练时，这张表也会被更新。训练久了以后，相似 token 的向量往往会变得更接近。

## 5. Attention 是什么

Attention 可以理解成：

```text
当前位置应该看前文里的哪些位置？
```

例如：

```text
我把书放在桌子上，因为它很平。
```

模型看到“它”时，应该更关注“桌子”，而不是“书”。

Attention 里有三个向量：

- query：我现在想找什么？
- key：每个位置能被怎么找到？
- value：每个位置真正携带的信息。

query 和 key 算相似度，相似度越高，就越关注对应位置的 value。

## 6. 为什么要 causal mask

训练语言模型时，不能让模型偷看未来。

例如预测第三个 token 时，只能看第一个和第二个 token。

如果模型能看到答案，它训练 loss 会很低，但生成时就废了，因为真实生成没有未来答案可看。

代码里使用：

```python
is_causal=True
```

这会自动屏蔽未来位置。

## 7. MLP 是什么

Attention 负责让 token 之间交流。

MLP 负责让每个 token 自己内部做更复杂的变换。

一个 Transformer Block 通常就是：

```text
Attention + MLP
```

重复很多层后，模型就能逐步形成更复杂的表示。

## 8. loss 是什么

模型输出的是 logits，也就是每个 token 的分数。

如果词表大小是 8000，那么每个位置会输出 8000 个分数。

交叉熵 loss 会问：

```text
正确答案那个 token 的分数够高吗？
```

如果正确 token 分数低，loss 就高。

训练目标就是让 loss 下降。

## 9. backward 和 optimizer 在干什么

`loss.backward()`：

```text
计算每个参数对 loss 的影响方向。
```

`optimizer.step()`：

```text
根据梯度修改参数，让下次 loss 更低。
```

这就是训练。

## 10. 你应该怎么读代码

推荐顺序：

1. `docs/llm/from_zero_principles.md`
2. `src/foundation_models/llm/data.py`
3. `src/foundation_models/llm/model.py`
4. `scripts/llm/train.py`
5. `scripts/llm/generate.py`

第一次不要试图理解每个数学细节。先抓住主线：

```text
文本 -> token -> x/y -> 模型 -> logits -> loss -> 更新参数 -> 生成文本
```
