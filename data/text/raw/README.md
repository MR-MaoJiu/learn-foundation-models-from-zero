# 训练素材说明

这个目录放原始中文文本语料。

当前自带文件：

- `tiny_zh_corpus.txt`：原创中文小语料，只适合跑通流程。

可选生成文件：

- `open_zh_wikipedia.txt`：通过 `scripts/llm/download_open_chinese_corpus.py` 下载的小份开源中文维基百科文本。
- `open_zh_wikipedia.source.md`：对应语料的来源和许可说明。
- `daily_life_dialogue.txt`：通过 `scripts/llm/download_daily_dialogue_corpus.py` 下载和生成的日常生活对话语料。
- `daily_life_dialogue.source.md`：对应日常对话语料的来源和许可说明。

## 如何加入更多语料

你可以把更多 `.txt` 文件放到这个目录，然后在命令里传入多个 `--input`：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --out artifacts/llm/tokenizer --vocab-size 8000
python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

如果你想让模型更像日常聊天，可以先生成日常对话语料：

```bash
python scripts/llm/download_daily_dialogue_corpus.py --out data/text/raw/daily_life_dialogue.txt --sources synthetic --max-synthetic 30000
```

默认推荐 `--sources synthetic`，这样生成的是简体中文日常生活对话。
`dailydialog` 和 `everyday` 是英文开源对话源，如果你只想训练简体中文，就不要加它们。

生成器会混合多种对话逻辑：

- 普通交流：直接围绕日常事情问答。
- 计划安排：先明确目标，再拆步骤。
- 问题排查：先定位卡点，再给处理办法。
- 临时变更：用户改计划，助手更新安排。
- 情绪安抚：先回应焦虑，再给可执行建议。
- 选择比较：比较两个方案，再做决定。

然后把它加入 tokenizer 和训练数据：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/daily_life_dialogue.txt --out artifacts/llm/tokenizer --vocab-size 1024 --min-frequency 1
python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/daily_life_dialogue.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

## 好语料的特点

- 来源合法，明确允许使用。
- 文本干净，乱码少，重复少。
- 主题多样，表达自然。
- 不包含隐私、密码、身份证号、手机号等敏感信息。

坏语料会让模型学到坏习惯，例如重复、格式混乱、偏见、隐私泄漏。
