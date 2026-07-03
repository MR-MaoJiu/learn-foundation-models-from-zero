# 训练素材说明

这个目录放原始文本和聊天 SFT 语料。

源码中保留：

- `tiny_zh_corpus.txt`：很小的原创中文种子文本，只用于引导流程。
- `sft_chat_zh.jsonl`：中文 `messages` SFT 数据，用于训练聊天助手行为。

运行流程时自动生成：

- `pretrain_zh.txt`：预训练普通文本语料，由种子文本、SFT 转写文本和合成段落组合而成。
- `open_zh_wikipedia.txt`：可选下载的开源中文文本。
- `sft_open_*.jsonl`：可选导入的开源 SFT 数据。

这些生成文件默认不提交。

## 两类语料

预训练语料是普通 `.txt`：

```text
语言模型会根据上下文预测下一个 token。训练时，模型不断降低预测错误。
```

SFT 语料是 `messages` JSONL：

```json
{"messages":[{"role":"system","content":"你是一个中文 AI 助手。"},{"role":"user","content":"语言模型是什么？"},{"role":"assistant","content":"语言模型会根据前文 token 预测下一个 token。"}]}
```

预训练让模型学会续写文本；SFT 让模型学会按用户问题回答。

## 默认生成命令

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

一键流程 `scripts/llm/run_chat_workflow.py` 会自动执行这些步骤。

## 加入更多真实语料

可以继续加入更多 `.txt` 作为预训练语料：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/pretrain_zh.txt data/text/raw/your_corpus.txt data/text/raw/sft_chat_zh.jsonl --out artifacts/llm/tokenizer --vocab-size 8000
python scripts/llm/prepare_data.py --input data/text/raw/pretrain_zh.txt data/text/raw/your_corpus.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

聊天样本不要写成普通“用户：/助手：”长文本，请整理成 `messages` JSONL 后走 SFT。

## 好语料标准

- 来源合法，许可清楚。
- 文本干净，乱码少，重复少。
- 主题多样，表达自然。
- 不包含隐私、密码、身份证号、手机号等敏感信息。
