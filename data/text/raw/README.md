# 训练素材说明

这个目录放原始中文文本语料。

当前自带文件：

- `tiny_zh_corpus.txt`：原创中文小语料，只适合跑通流程。

可选生成文件：

- `open_zh_wikipedia.txt`：通过 `scripts/llm/download_open_chinese_corpus.py` 下载的小份开源中文维基百科文本。
- `open_zh_wikipedia.source.md`：对应语料的来源和许可说明。

## 如何加入更多语料

你可以把更多 `.txt` 文件放到这个目录，然后在命令里传入多个 `--input`：

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --out artifacts/llm/tokenizer --vocab-size 8000
python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/open_zh_wikipedia.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

## 好语料的特点

- 来源合法，明确允许使用。
- 文本干净，乱码少，重复少。
- 主题多样，表达自然。
- 不包含隐私、密码、身份证号、手机号等敏感信息。

坏语料会让模型学到坏习惯，例如重复、格式混乱、偏见、隐私泄漏。
