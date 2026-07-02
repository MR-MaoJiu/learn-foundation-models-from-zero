# 视觉模型实现导读

这一章对应项目里的视觉学习线。你可以一边打开代码，一边按下面顺序阅读。

## 1. 目录和文件

```text
configs/vision/
  mnist_cnn_debug.json              # 视觉模型训练配置

scripts/vision/
  download_open_vision_dataset.py   # 下载并整理 MNIST 开源图片数据
  train_classifier.py               # 训练 CNN 分类模型
  predict_image.py                  # 用训练好的模型预测单张图片
  count_params.py                   # 只统计模型参数量，不训练

src/foundation_models/vision/
  config.py                         # 读取和校验配置
  data.py                           # 把图片文件和 labels.csv 变成 PyTorch Dataset
  model.py                          # SmallCNN 模型结构
  utils.py                          # 随机种子、设备选择、保存模型等工具函数

data/vision/
  README.md                         # 视觉数据说明
```

## 2. 推荐阅读顺序

不要从训练脚本第一行硬啃。初学者更适合按“数据 -> 模型 -> 训练”的顺序读。

1. 先读 `configs/vision/mnist_cnn_debug.json`。
   你会看到训练要用哪些文件、batch size 多大、学习率是多少、训练多少轮。

2. 再读 `src/foundation_models/vision/data.py`。
   这里解释了如何从 `labels.csv` 找到图片路径，并把 PNG 图片变成模型能吃的张量。

3. 再读 `src/foundation_models/vision/model.py`。
   这里定义了 CNN 的层：卷积、激活、池化、分类头。

4. 最后读 `scripts/vision/train_classifier.py`。
   这个脚本把配置、数据、模型、loss、优化器串成完整训练流程。

## 3. 第一步：准备 MNIST 数据

运行：

```bash
python scripts/vision/download_open_vision_dataset.py
```

它会从 Hugging Face Datasets 下载 MNIST，然后整理成更容易理解的文件结构。

整理后大致长这样：

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

`labels.csv` 每一行包含：

```text
image,label
images/000000.png,5
images/000001.png,0
```

这比直接读二进制数据更适合学习，因为你可以直接打开图片看。

## 4. 第二步：统计参数量

运行：

```bash
python scripts/vision/count_params.py --config configs/vision/mnist_cnn_debug.json
```

这个命令不会训练，只会创建模型并统计可训练参数。

你可以用它确认模型规模是否适合自己的电脑。

## 5. 第三步：训练模型

运行：

```bash
python scripts/vision/train_classifier.py --config configs/vision/mnist_cnn_debug.json
```

训练时你会看到类似：

```text
epoch 0 step 0 train loss ...
epoch 0 val loss ... val acc ...
```

含义：

- `train loss`：训练集当前 batch 上的错误程度。
- `val loss`：验证集整体错误程度。
- `val acc`：验证集准确率。

如果训练正常，`val acc` 通常会逐渐升高。

## 6. 第四步：预测单张图片

训练完成后会保存：

```text
checkpoints/vision/mnist_cnn_debug/last.pt
```

你可以拿验证集里的一张图片测试：

```bash
python scripts/vision/predict_image.py ^
  --config configs/vision/mnist_cnn_debug.json ^
  --checkpoint checkpoints/vision/mnist_cnn_debug/last.pt ^
  --image data/vision/mnist/val/images/000000.png
```

如果你在 macOS 或 Linux 上，可以把换行符写成反斜杠：

```bash
python scripts/vision/predict_image.py \
  --config configs/vision/mnist_cnn_debug.json \
  --checkpoint checkpoints/vision/mnist_cnn_debug/last.pt \
  --image data/vision/mnist/val/images/000000.png
```

## 7. 你应该重点看懂的代码

`ImageClassificationDataset.__getitem__`：

它负责根据索引读取一张图片和一个标签。PyTorch 的 `DataLoader` 会反复调用它。

`SmallCNN.forward`：

它定义了一张图片从输入到输出 logits 的计算路径。

`evaluate`：

它只评估，不训练，所以用了 `torch.no_grad()`，这样可以减少内存占用。

`loss.backward()`：

它计算每个参数对 loss 的影响，也就是梯度。

`optimizer.step()`：

它根据梯度更新参数。

## 8. 常见问题

如果提示缺少 `datasets` 或 `PIL`，先安装依赖：

```bash
pip install -r requirements.txt
```

如果训练很慢，可以把配置里的 `batch_size` 调小，比如从 64 改成 32。

如果内存不足，把 `device` 改成 `cpu`，速度会慢一些，但更稳定。
