# 从 0 到部署：小型中文聊天 LLM

这个仓库只保留一条主线：**从零实现并训练一个小型中文聊天 LLM，然后完成 SFT、评估、导出和本地 HTTP 部署**。

标准流程是：

```text
准备语料 -> 训练 tokenizer -> 预训练 base model -> SFT chat model -> 评估 -> 导出 -> HTTP 部署
```

也就是说：**先预训练，再 SFT**。预训练让模型学会续写和基本语言分布；SFT 让模型学会按照用户问题输出助手回复。跳过预训练直接 SFT 只适合测试代码链路，不适合追求正常聊天质量。

## 当前流程

`scripts/llm/run_chat_workflow.py` 会按下面顺序执行：

1. 生成中文 `messages` SFT 数据。
2. 校验 SFT 数据格式。
3. 构建普通文本预训练语料 `pretrain_zh.txt`。
4. 训练 tokenizer。
5. 把预训练文本转换成 `train.bin` / `val.bin`。
6. 预训练 base model。
7. 基于 base checkpoint 做 SFT。
8. 评估 SFT checkpoint。
9. 导出部署包。

聊天数据使用 OpenAI/GPT 常见的 `messages` JSONL 格式，内部统一渲染成 ChatML：

```text
<|system|>
系统指令
<|end|>
<|user|>
用户问题
<|end|>
<|assistant|>
助手回答
<|end|>
```

## 目录结构

```text
configs/llm/
  debug.json                 # debug 预训练配置
  sft_debug.json             # debug SFT 配置
  gpt_50m_8gb.json           # 50M 档预训练配置
  sft_50m_8gb.json           # 50M 档 SFT 配置

data/text/raw/
  tiny_zh_corpus.txt         # 很小的种子文本
  sft_chat_zh.jsonl          # 中文 messages SFT 数据
  README.md                  # 数据说明

scripts/llm/
  build_pretrain_corpus.py
  build_sft_chat_corpus.py
  validate_sft_data.py
  train_tokenizer.py
  prepare_data.py
  train.py
  train_sft.py
  evaluate_chat.py
  export_model.py
  serve_chat.py
  run_chat_workflow.py

src/foundation_models/llm/
  tokenizer.py
  data.py
  model.py
  chat.py
  sft_data.py
  config.py
  utils.py
```

运行后会生成这些可重建产物，默认不提交：

```text
artifacts/
checkpoints/
data/text/raw/pretrain_zh.txt
data/text/processed/
runs/
deployments/
```

## 安装

```bash
cd learn-foundation-models-from-zero
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

Windows + NVIDIA GPU 可以安装 CUDA 版 PyTorch：

```bash
python -m pip install -r requirements-cuda.txt
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 一键运行

debug 流程用于快速验证代码链路：

```bash
python scripts/llm/run_chat_workflow.py
```

debug 模型很小，只适合确认流程，不要期待它达到主流模型聊天质量。

更接近真实训练的 50M 档位：

```bash
python scripts/llm/run_chat_workflow.py --pretrain-config configs/llm/gpt_50m_8gb.json --sft-config configs/llm/sft_50m_8gb.json --export-dir deployments/llm/chat_50m_8gb --vocab-size 2048
```

复用已有 SFT checkpoint，只重新评估和导出：

```bash
python scripts/llm/run_chat_workflow.py --skip-pretrain --skip-sft
```

`--skip-pretrain` 会复用已有 tokenizer 和 base checkpoint，不会重新训练 tokenizer，避免产物不匹配。

## 分步运行

生成 SFT 数据：

```bash
python scripts/llm/build_sft_chat_corpus.py --out data/text/raw/sft_chat_zh.jsonl --count 10000 --seed 42
```

校验 SFT：

```bash
python scripts/llm/validate_sft_data.py --input data/text/raw/sft_chat_zh.jsonl --show-template
```

构建预训练语料：

```bash
python scripts/llm/build_pretrain_corpus.py --out data/text/raw/pretrain_zh.txt --seed-text data/text/raw/tiny_zh_corpus.txt --sft-jsonl data/text/raw/sft_chat_zh.jsonl --count 8000 --max-sft-rows 10000 --seed 42
```

训练 tokenizer：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/pretrain_zh.txt data/text/raw/sft_chat_zh.jsonl --out artifacts/llm/tokenizer --vocab-size 2048 --min-frequency 1
```

准备预训练数据：

```bash
python scripts/llm/prepare_data.py --input data/text/raw/pretrain_zh.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

预训练：

```bash
python scripts/llm/train.py --config configs/llm/debug.json
```

SFT：

```bash
python scripts/llm/train_sft.py --config configs/llm/sft_debug.json
```

评估：

```bash
python scripts/llm/evaluate_chat.py --checkpoint checkpoints/llm/sft_debug/last.pt
```

导出：

```bash
python scripts/llm/export_model.py --checkpoint checkpoints/llm/sft_debug/last.pt --out deployments/llm/chat_debug
```

启动服务：

```bash
python scripts/llm/serve_chat.py --model-dir deployments/llm/chat_debug
```

## 详细文档

- [LLM 原理](docs/llm/from_zero_principles.md)
- [代码导读](docs/llm/implementation_guide.md)
- [完整训练到部署流程](docs/llm/full_workflow.md)
