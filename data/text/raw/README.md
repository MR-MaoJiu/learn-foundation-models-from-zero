# 训练素材说明

这个目录放原始中文文本语料。

当前自带文件：

- `tiny_zh_corpus.txt`：原创中文小语料，只适合跑通流程。
- `sft_chat_zh.jsonl`：原创中文 SFT 样例数据，采用 OpenAI/GPT 常见的 `messages` JSONL 格式，用来模拟“用户问题 -> 助手回答”的监督微调流程。

可选生成文件：

- `open_zh_wikipedia.txt`：通过 `scripts/llm/download_open_chinese_corpus.py` 下载的小份开源中文维基百科文本。
- `open_zh_wikipedia.source.md`：对应语料的来源和许可说明。

## 如何加入更多语料

你可以把更多 `.txt` 文件放到这个目录，然后在命令里传入多个 `--input`：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --out artifacts/llm/tokenizer --vocab-size 8000
python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

聊天能力不要靠普通 `.txt` 对话转写来训练。请把聊天样本整理成下面的 `messages` JSONL，然后通过 SFT 训练。

## 好语料的特点

- 来源合法，明确允许使用。
- 文本干净，乱码少，重复少。
- 主题多样，表达自然。
- 不包含隐私、密码、身份证号、手机号等敏感信息。

坏语料会让模型学到坏习惯，例如重复、格式混乱、偏见、隐私泄漏。

## SFT 聊天语料格式

推荐每行使用 `messages`：

```json
{"messages":[{"role":"system","content":"你是一个中文 AI 助手。回答要准确、自然、简洁。"},{"role":"user","content":"语言模型是什么？"},{"role":"assistant","content":"语言模型会根据前文 token 预测下一个 token。"}]}
```

格式要求：

- `system` 最多一条，只能放第一条。
- 去掉 `system` 后，`user` 和 `assistant` 必须交替。
- 最后一条必须是 `assistant`，因为它是当前样本的训练目标。
- 不要把整段对话写成一个普通字符串；要拆成多条 role/content 消息。

训练前检查格式：

```bash
python scripts/llm/validate_sft_data.py --input data/text/raw/sft_chat_zh.jsonl --show-template
```

脚本会把样本渲染成 ChatML 预览：

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
