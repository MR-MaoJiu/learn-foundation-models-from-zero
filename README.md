# 从 0 学习基础模型

`learn-foundation-models-from-zero` 是一个面向初学者的基础模型学习项目。

它现在包含三条学习线：

- `llm`：文本语言模型，学习 tokenizer、Transformer、next-token prediction。
- `vision`：视觉模型，学习图片张量、CNN、图像分类。
- `multimodal`：多模态模型，学习图像编码、文本编码、图文对齐。

所有模块都尽量使用清晰的小模型和开源学习素材，先让你理解主线，再逐步扩大规模。

## 开源训练素材

默认数据源：

- 文本：`wikimedia/wikipedia` 中文子集。Hugging Face 数据集卡说明 Wikipedia 原始文本使用 GFDL 和 CC BY-SA 3.0。
- 视觉：`ylecun/mnist`。Hugging Face 数据集卡说明 MNIST 有 70,000 张 28x28 手写数字图，许可信息为 MIT Licence。
- 多模态：使用 MNIST 图片，并由本项目根据标签生成文字描述，构造图文配对。

数据源页面：

- https://huggingface.co/datasets/wikimedia/wikipedia
- https://huggingface.co/datasets/ylecun/mnist

## 目录结构

```text
learn-foundation-models-from-zero/
├─ configs/
│  ├─ llm/
│  │  ├─ debug.json                  # LLM 入门调试配置：模型很小，适合先跑通流程
│  │  └─ gpt_50m_8gb.json            # 稍大一点的 LLM 配置：理解扩展规模时参考
│  ├─ vision/
│  │  └─ mnist_cnn_debug.json        # 视觉 CNN 配置：训练 MNIST 手写数字分类
│  └─ multimodal/
│     └─ mnist_clip_debug.json       # 多模态 TinyCLIP 配置：训练 MNIST 图文对齐
│
├─ data/
│  ├─ text/
│  │  └─ raw/
│  │     ├─ README.md                # 文本训练素材说明
│  │     └─ tiny_zh_corpus.txt       # 内置极小中文语料，不下载也能先跑通
│  ├─ vision/
│  │  └─ README.md                   # 视觉训练素材说明，MNIST 会下载到这里
│  └─ multimodal/
│     └─ README.md                   # 多模态训练素材说明，图文对 JSONL 会生成到这里
│
├─ docs/
│  ├─ README.md                      # 学习文档导航：建议先读什么、后读什么
│  ├─ llm/
│  │  ├─ from_zero_principles.md     # LLM 原理：token、embedding、Transformer、预测下一个 token
│  │  └─ implementation_guide.md     # LLM 代码导读：按文件解释实现流程
│  ├─ vision/
│  │  ├─ from_zero_principles.md     # 视觉模型原理：图片张量、卷积、池化、分类 loss
│  │  └─ implementation_guide.md     # 视觉代码导读：MNIST 数据、CNN、训练循环
│  └─ multimodal/
│     ├─ from_zero_principles.md     # 多模态原理：图片向量、文本向量、对比学习
│     └─ implementation_guide.md     # 多模态代码导读：图文对、TinyCLIP、相似度矩阵
│
├─ scripts/
│  ├─ llm/
│  │  ├─ download_open_chinese_corpus.py  # 下载开源中文语料，保存成普通 txt
│  │  ├─ train_tokenizer.py               # 训练 tokenizer，把文本切成 token
│  │  ├─ prepare_data.py                  # 把文本转成 token id，并切分 train/val
│  │  ├─ train.py                         # 训练 GPT 风格小语言模型
│  │  ├─ generate.py                      # 加载 checkpoint，根据 prompt 生成文本
│  │  └─ count_params.py                  # 统计 LLM 参数量，不训练
│  ├─ vision/
│  │  ├─ download_open_vision_dataset.py  # 下载 MNIST，并导出 PNG 图片和 labels.csv
│  │  ├─ train_classifier.py              # 训练 CNN 手写数字分类模型
│  │  ├─ predict_image.py                 # 用训练好的 CNN 预测单张图片
│  │  └─ count_params.py                  # 统计视觉模型参数量，不训练
│  └─ multimodal/
│     ├─ build_mnist_image_text_pairs.py  # 根据 MNIST 标签生成中文描述，构造图文对
│     ├─ train_contrastive.py             # 训练 TinyCLIP 图文对齐模型
│     ├─ rank_texts_for_image.py          # 给一张图片，从多条文本里找最匹配的
│     └─ count_params.py                  # 统计多模态模型参数量，不训练
│
├─ src/
│  └─ foundation_models/
│     ├─ llm/
│     │  ├─ config.py                 # LLM 配置结构和 JSON 读取
│     │  ├─ tokenizer.py              # tokenizer 训练和加载相关函数
│     │  ├─ data.py                   # LLM 数据读取：把 token id 喂给模型
│     │  ├─ model.py                  # GPT/Transformer 模型主体
│     │  └─ utils.py                  # 随机种子、设备选择、checkpoint 保存等工具
│     ├─ vision/
│     │  ├─ config.py                 # 视觉模型配置结构和 JSON 读取
│     │  ├─ data.py                   # 图片分类 Dataset：读取 PNG 和 labels.csv
│     │  ├─ model.py                  # SmallCNN 模型结构
│     │  └─ utils.py                  # 视觉训练用工具函数
│     └─ multimodal/
│        ├─ config.py                 # 多模态配置结构和 JSON 读取
│        ├─ data.py                   # 图文对 Dataset 和字符级 tokenizer
│        ├─ model.py                  # TinyCLIP：图片编码器、文本编码器、对比学习 loss
│        └─ utils.py                  # 多模态训练用工具函数
│
├─ .gitignore
├─ README.md
└─ requirements.txt
```

你可以这样理解这些目录：

- `configs/`：只放配置，不放训练逻辑。想调模型大小、学习率、batch size，优先改这里。
- `data/`：只放训练素材和数据说明。脚本下载或生成的数据也会放在这里。
- `docs/`：放原理和代码导读。看不懂代码时，先读这里。
- `scripts/`：放可以直接运行的命令行脚本。它们负责“把一件事跑起来”。
- `src/`：放真正的 Python 模块。模型结构、数据集类、工具函数都在这里。

`scripts/` 和 `src/` 的区别很重要：

```text
scripts/ = 入口脚本，负责串流程，例如下载数据、训练模型、生成文本
src/     = 可复用代码，负责具体能力，例如模型结构、数据读取、配置解析
```

举例：

```text
你运行 scripts/vision/train_classifier.py
它会调用 src/foundation_models/vision/data.py 读取图片
它会调用 src/foundation_models/vision/model.py 创建 CNN
它会调用 src/foundation_models/vision/utils.py 保存模型
```

运行后会生成这些目录或文件：

```text
artifacts/
checkpoints/
data/text/processed/
data/text/raw/open_zh_wikipedia.txt
data/vision/mnist/
data/multimodal/mnist_pairs/
```

## 安装

```bash
cd learn-foundation-models-from-zero
python -m venv .venv
```

激活虚拟环境：

```bash
# macOS / Linux
source .venv/bin/activate
```

```bat
:: Windows
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你的系统里 `python` 命令不可用，可以把下面命令里的 `python` 换成 `python3`。

## 学习线 1：文本 LLM

先读：

- [docs/llm/from_zero_principles.md](docs/llm/from_zero_principles.md)
- [docs/llm/implementation_guide.md](docs/llm/implementation_guide.md)

可选下载开源中文语料：

```bash
python scripts/llm/download_open_chinese_corpus.py --out data/text/raw/open_zh_wikipedia.txt --max-docs 200
```

训练 tokenizer：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt --out artifacts/llm/tokenizer --vocab-size 2000
```

准备训练数据：

```bash
python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

训练 debug 小模型：

```bash
python scripts/llm/train.py --config configs/llm/debug.json
```

生成文本：

```bash
python scripts/llm/generate.py --checkpoint checkpoints/llm/debug/last.pt --prompt "语言模型的目标是"
```

## 学习线 2：视觉模型

先读：

- [docs/vision/from_zero_principles.md](docs/vision/from_zero_principles.md)
- [docs/vision/implementation_guide.md](docs/vision/implementation_guide.md)

下载 MNIST：

```bash
python scripts/vision/download_open_vision_dataset.py --out data/vision/mnist --max-train 2000 --max-val 500
```

统计 CNN 参数：

```bash
python scripts/vision/count_params.py --config configs/vision/mnist_cnn_debug.json
```

训练 CNN：

```bash
python scripts/vision/train_classifier.py --config configs/vision/mnist_cnn_debug.json
```

预测单张图片：

```bash
python scripts/vision/predict_image.py --checkpoint checkpoints/vision/mnist_cnn_debug/last.pt --image data/vision/mnist/val/images/000000.png
```

## 学习线 3：多模态图文对齐

先读：

- [docs/multimodal/from_zero_principles.md](docs/multimodal/from_zero_principles.md)
- [docs/multimodal/implementation_guide.md](docs/multimodal/implementation_guide.md)

多模态模块依赖视觉模块导出的 MNIST 图片。如果还没有下载 MNIST，先运行：

```bash
python scripts/vision/download_open_vision_dataset.py --out data/vision/mnist --max-train 2000 --max-val 500
```

构建 MNIST 图文配对：

```bash
python scripts/multimodal/build_mnist_image_text_pairs.py --mnist-dir data/vision/mnist --out data/multimodal/mnist_pairs
```

统计 TinyCLIP 参数：

```bash
python scripts/multimodal/count_params.py --config configs/multimodal/mnist_clip_debug.json
```

训练图文对齐模型：

```bash
python scripts/multimodal/train_contrastive.py --config configs/multimodal/mnist_clip_debug.json
```

给图片匹配文本：

```bash
python scripts/multimodal/rank_texts_for_image.py --checkpoint checkpoints/multimodal/mnist_clip_debug/last.pt --image data/vision/mnist/val/images/000000.png --texts "这是一张数字 0 的手写图片。" "这是一张数字 7 的手写图片。"
```

## 总学习顺序

建议先这样走：

1. 文本 LLM：理解 token、Transformer、next-token loss。
2. 视觉 CNN：理解图片张量、卷积、分类 loss。
3. 多模态 TinyCLIP：理解图像向量、文本向量、对比学习。

总主线：

```text
单模态表示学习 -> 跨模态表示对齐 -> 更大的基础模型
```
