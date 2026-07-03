# LLM 从训练到部署完整流程

本项目采用标准的聊天模型训练链路：

```text
数据准备 -> tokenizer -> 预训练 base model -> SFT chat model -> 评估 -> 导出 -> HTTP 部署
```

结论很明确：**正常流程是先预训练，再 SFT**。

- 预训练：让模型学习通用文本分布和续写能力。
- SFT：让模型学习 `system/user/assistant` 结构下的助手回答方式。
- 部署：加载 SFT 后的 chat checkpoint，而不是 base checkpoint。

跳过预训练直接 SFT 只适合调试代码，不适合追求正常聊天效果。

## 1. 语料准备

本项目有两类语料。

预训练语料是普通文本：

```text
data/text/raw/pretrain_zh.txt
```

SFT 语料是 `messages` JSONL：

```text
data/text/raw/sft_chat_zh.jsonl
```

生成 SFT：

```bash
python scripts/llm/build_sft_chat_corpus.py --out data/text/raw/sft_chat_zh.jsonl --count 10000 --seed 42
```

校验 SFT：

```bash
python scripts/llm/validate_sft_data.py --input data/text/raw/sft_chat_zh.jsonl --show-template
```

生成预训练文本：

```bash
python scripts/llm/build_pretrain_corpus.py --out data/text/raw/pretrain_zh.txt --seed-text data/text/raw/tiny_zh_corpus.txt --sft-jsonl data/text/raw/sft_chat_zh.jsonl --count 8000 --max-sft-rows 10000 --seed 42
```

`pretrain_zh.txt` 是可重建产物，默认不提交。它解决了 tiny 语料太少导致 `val.bin` token 数不足的问题。

## 2. 训练 tokenizer

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/pretrain_zh.txt data/text/raw/sft_chat_zh.jsonl --out artifacts/llm/tokenizer --vocab-size 2048 --min-frequency 1
```

验收标准：

- 生成 `artifacts/llm/tokenizer/tokenizer.json`。
- tokenizer 包含 `<|system|>`、`<|user|>`、`<|assistant|>`、`<|end|>`。
- 后续预训练、SFT、评估和部署都使用同一个 tokenizer。

## 3. 准备预训练数据

```bash
python scripts/llm/prepare_data.py --input data/text/raw/pretrain_zh.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

产物：

```text
data/text/processed/train.bin
data/text/processed/val.bin
```

这一步只使用普通 `.txt` 预训练语料。SFT 的 `messages` JSONL 不直接放进 `train.bin`。

## 4. 预训练 base model

debug 配置：

```bash
python scripts/llm/train.py --config configs/llm/debug.json
```

产物：

```text
checkpoints/llm/debug/last.pt
```

50M 档配置：

```bash
python scripts/llm/train.py --config configs/llm/gpt_50m_8gb.json
```

产物：

```text
checkpoints/llm/gpt_50m_8gb/last.pt
```

base model 主要是续写模型，不应该直接作为聊天助手部署。

## 5. SFT 成 chat model

debug SFT：

```bash
python scripts/llm/train_sft.py --config configs/llm/sft_debug.json
```

它会读取：

```text
checkpoints/llm/debug/last.pt
```

并输出：

```text
checkpoints/llm/sft_debug/last.pt
```

50M 档 SFT：

```bash
python scripts/llm/train_sft.py --config configs/llm/sft_50m_8gb.json
```

它会读取：

```text
checkpoints/llm/gpt_50m_8gb/last.pt
```

并输出：

```text
checkpoints/llm/sft_50m_8gb/last.pt
```

SFT 的关键点：

```text
prompt 部分：labels = -100，不参与 loss
最后 assistant 回复：参与 loss
<|end|>：参与 loss，让模型学会停止当前轮回复
```

## 6. 评估

debug：

```bash
python scripts/llm/evaluate_chat.py --checkpoint checkpoints/llm/sft_debug/last.pt
```

50M：

```bash
python scripts/llm/evaluate_chat.py --checkpoint checkpoints/llm/sft_50m_8gb/last.pt
```

当前评估是烟雾测试，检查：

- 回复非空。
- 没有明显泄露 `<|user|>`、`<|system|>` 等角色 token。
- 回复长度没有异常。

真实项目还应加入人工评估、事实性评估、安全评估、延迟评估和成本评估。

## 7. 导出部署包

debug：

```bash
python scripts/llm/export_model.py --checkpoint checkpoints/llm/sft_debug/last.pt --out deployments/llm/chat_debug
```

50M：

```bash
python scripts/llm/export_model.py --checkpoint checkpoints/llm/sft_50m_8gb/last.pt --out deployments/llm/chat_50m_8gb
```

部署包包含：

```text
model.pt
tokenizer.json
manifest.json
```

## 8. 启动服务

```bash
python scripts/llm/serve_chat.py --model-dir deployments/llm/chat_debug --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

聊天请求：

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"语言模型是什么？\"}"
```

## 9. 一键流程

debug 流程：

```bash
python scripts/llm/run_chat_workflow.py
```

50M 档：

```bash
python scripts/llm/run_chat_workflow.py --pretrain-config configs/llm/gpt_50m_8gb.json --sft-config configs/llm/sft_50m_8gb.json --export-dir deployments/llm/chat_50m_8gb --vocab-size 2048
```

一键脚本会根据 `--sft-config` 的 `training.output_dir` 自动选择要评估和导出的 SFT checkpoint。

复用已有 base checkpoint，只重新做 SFT：

```bash
python scripts/llm/run_chat_workflow.py --skip-pretrain
```

复用已有 SFT checkpoint，只重新评估和导出：

```bash
python scripts/llm/run_chat_workflow.py --skip-pretrain --skip-sft
```

注意：`--skip-pretrain` 会复用已有 tokenizer 和 base checkpoint，不会重新训练 tokenizer 或重建预训练 `.bin`。这样可以避免 checkpoint 和 tokenizer 不匹配。

## 10. 常见问题

### 为什么不是直接 SFT？

可以直接 SFT，但那是从随机初始化开始学，会很不稳定，也很难得到正常聊天能力。标准流程是先预训练得到 base model，再用 SFT 对齐成 chat model。

### 为什么 debug 模型聊天质量一般？

debug 模型参数量很小，训练步数也少。它的目标是跑通工程链路，不是达到主流模型效果。想要更像正常助手，使用 50M 档，并继续增加真实高质量语料和训练步数。

### tokenizer、base checkpoint、SFT checkpoint 能混用吗？

不能。它们必须来自同一条训练链路。换 tokenizer 后，必须重新准备 `.bin`，重新预训练，再重新 SFT。

### `val.bin has 185 tokens` 怎么办？

说明预训练语料太少。先生成 `pretrain_zh.txt`，再重新运行 `prepare_data.py`。当前一键流程已经自动处理这个问题。
