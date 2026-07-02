# 多模态训练素材

这个目录用于保存图文对数据。

当前项目用 MNIST 图片和中文描述构造最小多模态数据集。这样可以在不引入复杂大数据的情况下，先学会 CLIP 风格图文对齐的核心原理。

## 数据生成脚本

先准备视觉数据：

```bash
python scripts/vision/download_open_vision_dataset.py
```

再生成图文对：

```bash
python scripts/multimodal/build_mnist_image_text_pairs.py
```

生成后目录大致是：

```text
data/multimodal/mnist_pairs/
  SOURCE.md
  train.jsonl
  val.jsonl
```

## JSONL 是什么

JSONL 表示每一行都是一个 JSON 对象。

示例：

```json
{"image":"data/vision/mnist/train/images/000000.png","text":"这是一张手写数字五。","label":5}
```

相比一个巨大的 JSON 数组，JSONL 更适合训练数据，因为程序可以一行一行读取。

## 为什么用 MNIST 做多模态

真实多模态数据通常来自网页、图文说明、图片标题等，数据清洗会比较复杂。

这个学习项目先用可控的小数据，让你清楚看到：

- 图片来自哪里。
- 文本如何生成。
- 哪些图文是正确配对。
- 模型如何学习相似度。

理解这条小链路后，再学习更大的图文数据会容易很多。
