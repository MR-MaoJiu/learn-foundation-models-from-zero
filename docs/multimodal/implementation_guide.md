# 多模态模型实现导读

这一章帮助你读懂 TinyCLIP 的实现。

你不需要先懂大型多模态模型。先看懂这个最小版本，就能理解很多大模型论文里的关键词：image encoder、text encoder、embedding、contrastive learning、similarity matrix。

## 1. 目录和文件

```text
configs/multimodal/
  mnist_clip_debug.json                # 多模态训练配置

scripts/multimodal/
  build_mnist_image_text_pairs.py      # 把 MNIST 图片和中文描述配成图文对
  train_contrastive.py                 # 训练 TinyCLIP
  rank_texts_for_image.py              # 给一张图片，从多句话里找最匹配的
  count_params.py                      # 统计模型参数量

src/foundation_models/multimodal/
  config.py                            # 配置读取和校验
  data.py                              # JSONL 图文对数据集、字符 tokenizer
  model.py                             # TinyCLIP 模型结构和对比学习 loss
  utils.py                             # 通用工具函数

data/multimodal/
  README.md                            # 多模态数据说明
```

## 2. 数据从哪里来

多模态训练需要图文对。

为了从 0 学习，本项目不一开始就引入复杂网页图文数据，而是先复用 MNIST：

```text
图片：data/vision/mnist/train/images/000001.png
文本：这是一张手写数字五。
```

这样你可以清楚知道：

- 图片是什么。
- 文本为什么匹配这张图片。
- 错误匹配是什么样子。

## 3. 第一步：先准备视觉数据

多模态脚本依赖 MNIST 图片，所以先运行：

```bash
python scripts/vision/download_open_vision_dataset.py
```

如果已经下载过，就不用重复下载。

## 4. 第二步：构造图文对

运行：

```bash
python scripts/multimodal/build_mnist_image_text_pairs.py
```

它会读取：

```text
data/vision/mnist/train/labels.csv
data/vision/mnist/val/labels.csv
```

然后生成：

```text
data/multimodal/mnist_pairs/train.jsonl
data/multimodal/mnist_pairs/val.jsonl
```

JSONL 是一种适合训练数据的格式，每一行都是一个独立 JSON。

示例：

```json
{"image":"data/vision/mnist/train/images/000001.png","text":"这是一张手写数字五。","label":5}
```

## 5. 第三步：统计参数量

运行：

```bash
python scripts/multimodal/count_params.py --config configs/multimodal/mnist_clip_debug.json
```

它会告诉你 TinyCLIP 有多少可训练参数。

## 6. 第四步：训练 TinyCLIP

运行：

```bash
python scripts/multimodal/train_contrastive.py --config configs/multimodal/mnist_clip_debug.json
```

训练时你会看到：

```text
epoch 0 step 0 train loss ...
epoch 0 val loss ...
```

这里的 loss 不是分类 loss，而是对比学习 loss。

loss 越低，表示正确图文对的相似度越容易排到前面。

## 7. 第五步：测试图文匹配

训练完成后，运行：

```bash
python scripts/multimodal/rank_texts_for_image.py ^
  --config configs/multimodal/mnist_clip_debug.json ^
  --checkpoint checkpoints/multimodal/mnist_clip_debug/last.pt ^
  --image data/vision/mnist/val/images/000000.png ^
  --texts "这是一张手写数字零。" "这是一张手写数字一。" "这是一张手写数字七。"
```

macOS 或 Linux 可以写成：

```bash
python scripts/multimodal/rank_texts_for_image.py \
  --config configs/multimodal/mnist_clip_debug.json \
  --checkpoint checkpoints/multimodal/mnist_clip_debug/last.pt \
  --image data/vision/mnist/val/images/000000.png \
  --texts "这是一张手写数字零。" "这是一张手写数字一。" "这是一张手写数字七。"
```

脚本会按相似度从高到低输出文本。

## 8. 你应该重点看懂的代码

`CharTokenizer`：

它把中文句子拆成字符，并把字符转成编号。这里不是工业级 tokenizer，只是为了让你看懂文本如何进入神经网络。

`ImageTextDataset.__getitem__`：

它一次返回三样东西：

```text
图片张量 images
文本编号 token_ids
文本 mask attention_mask
```

`ImageEncoder`：

它把图片变成向量。

`TextEncoder`：

它把文本变成向量。

`TinyCLIP.forward`：

这是核心。它会：

1. 编码图片。
2. 编码文本。
3. 计算相似度矩阵。
4. 构造正确配对标签。
5. 计算图片找文字和文字找图片两个方向的 loss。

## 9. 常见问题

如果提示找不到 MNIST 图片，先运行视觉数据下载脚本。

如果训练效果不好，可以多训练几轮，或者调大 `embed_dim`。

如果你只是想理解原理，不需要追求高准确率。先把每一步输入输出形状看懂，比盲目调参更重要。
