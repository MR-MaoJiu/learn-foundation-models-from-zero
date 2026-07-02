# 视觉训练素材

这个目录用于保存视觉模型学习用的数据。

当前项目使用 MNIST 手写数字数据集。它很小、下载快、图片简单，适合初学者理解视觉模型训练流程。

## 数据来源

下载脚本：

```text
scripts/vision/download_open_vision_dataset.py
```

默认使用 Hugging Face 上的 `ylecun/mnist` 数据集。

下载后脚本会在本目录生成：

```text
data/vision/mnist/
  SOURCE.md
  train/
    labels.csv
    images/
      000000.png
      000001.png
      ...
  val/
    labels.csv
    images/
      000000.png
      000001.png
      ...
```

这些生成文件没有提交到项目里，因为它们可以通过脚本重新获得。

## labels.csv 是什么

`labels.csv` 记录图片路径和标签：

```text
image,label
images/000000.png,5
images/000001.png,0
```

训练脚本会读取它，然后找到对应图片。

## 为什么这样整理

很多机器学习数据集默认格式不适合初学者直接观察。

本项目把 MNIST 导出成 PNG 图片和 CSV 标签，是为了让你可以直接打开图片，确认模型看到的训练素材到底是什么。
