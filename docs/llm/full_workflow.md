# LLM 从训练到部署完整工作流

这份文档把本项目的 LLM 线整理成一个真实工程的缩小版。

目标不是训练出强大的商业模型，而是完整模拟工作方式：

```text
数据准备 -> tokenizer -> 预训练 -> SFT -> 评估 -> 导出 -> HTTP 部署 -> 接口测试
```

本流程的聊天数据采用 OpenAI/GPT 常见的 `messages` JSONL 格式，内部训练模板采用 ChatML 风格特殊 token：

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

不要把新 SFT 数据和旧 checkpoint 混用。tokenizer、base checkpoint、SFT checkpoint、部署包必须属于同一条训练链路。

## 0. 工作流产物

完整跑完后，你会看到这些关键产物：

```text
artifacts/llm/tokenizer/tokenizer.json      # tokenizer
data/text/processed/train.bin               # 预训练训练集 token
data/text/processed/val.bin                 # 预训练验证集 token
checkpoints/llm/debug/last.pt               # base checkpoint
checkpoints/llm/sft_debug/last.pt           # chat checkpoint
runs/llm/chat_eval.jsonl                    # 冒烟评估结果
deployments/llm/chat_debug/                 # 部署包
```

真实项目里也会有类似边界：

- `data/`：数据资产。
- `artifacts/`：中间产物。
- `checkpoints/`：训练产物。
- `runs/`：实验日志和评估结果。
- `deployments/`：可部署交付物。

## 1. 准备语料

预训练语料用于学习语言规律，SFT 语料用于学习助手回答方式。

本仓库内置：

```text
data/text/raw/tiny_zh_corpus.txt
data/text/raw/sft_chat_zh.jsonl
```

默认 SFT 文件可以由脚本重建：

```bash
python scripts/llm/build_sft_chat_corpus.py --out data/text/raw/sft_chat_zh.jsonl --count 2000 --seed 42
```

这会生成约 2000 条中文 `messages` 样本，覆盖：

- 日常沟通：购物、做饭、通勤、睡眠、旅行、家庭沟通。
- 学习解释：tokenizer、embedding、attention、loss、SFT、checkpoint。
- 工程排查：CUDA、NaN、checkpoint/tokenizer 不匹配、重复生成。
- 写作润色：通知、礼貌表达、总结改写。
- 计划安排：学习、技术分享、训练小模型、找工作。
- 安全边界：实时天气、医疗、金融、隐私和不确定信息。
- 多轮上下文：至少两轮以上的追问和承接。

推荐 SFT JSONL 每行长这样：

```json
{"messages":[{"role":"system","content":"你是一个中文 AI 助手。回答要准确、自然、简洁。"},{"role":"user","content":"语言模型是什么？"},{"role":"assistant","content":"语言模型会根据前文 token 预测下一个 token，并通过训练逐步降低预测错误。"}]}
```

格式要求：

- `messages` 是列表。
- `role` 只允许 `system`、`user`、`assistant`。
- `system` 最多一条，只能放第一条。
- 去掉 `system` 后，`user` 和 `assistant` 必须交替。
- 每行最后一条必须是 `assistant`，因为它是 SFT 的训练目标。

训练前先检查 SFT 格式：

```bash
python scripts/llm/validate_sft_data.py --input data/text/raw/sft_chat_zh.jsonl --show-template
```

可选导入开源英文日常对话：

```bash
python scripts/llm/import_open_sft_corpus.py --source everyday --out data/text/raw/sft_open_everyday.jsonl --max-rows 2000
python scripts/llm/validate_sft_data.py --input data/text/raw/sft_open_everyday.jsonl --show-template
```

当前支持 `HuggingFaceTB/everyday-conversations-llama3.1-2k`。它的 Hugging Face 页面展示的是多轮 role/content messages，覆盖 everyday topics 和 basic science。使用前请按你的用途再次确认数据集页面的许可和限制。

真实工作里，这一步还会做数据清洗、去重、敏感信息过滤、许可证检查和数据版本记录。

## 2. 训练 tokenizer

```bash
python scripts/llm/train_tokenizer.py --input data/text/raw/tiny_zh_corpus.txt data/text/raw/sft_chat_zh.jsonl --out artifacts/llm/tokenizer --vocab-size 1024 --min-frequency 1
```

验收标准：

- 生成 `artifacts/llm/tokenizer/tokenizer.json`。
- 后续预训练、SFT、生成、部署都使用同一个 tokenizer。
- tokenizer 里包含 `<|system|>`、`<|user|>`、`<|assistant|>`、`<|end|>` 这些聊天特殊 token。

## 3. 准备预训练数据

```bash
python scripts/llm/prepare_data.py --input data/text/raw/tiny_zh_corpus.txt --tokenizer artifacts/llm/tokenizer/tokenizer.json --out data/text/processed --val-ratio 0.1
```

验收标准：

- 生成 `data/text/processed/train.bin`。
- 生成 `data/text/processed/val.bin`。

注意：SFT 的 JSONL 不放进预训练 bin。预训练学“文本分布”，SFT 学“助手回答”。

## 4. 预训练 base model

```bash
python scripts/llm/train.py --config configs/llm/debug.json
```

产物：

```text
checkpoints/llm/debug/last.pt
```

验收标准：

- train loss 能正常下降。
- checkpoint 可以被 `generate.py` 加载。

快速试生成：

```bash
python scripts/llm/generate.py --checkpoint checkpoints/llm/debug/last.pt --prompt "语言模型是什么？"
```

这个阶段的模型主要是“续写模型”，不一定像助手。

## 5. SFT 成 chat model

```bash
python scripts/llm/train_sft.py --config configs/llm/sft_debug.json
```

如果你导入了开源数据，可以在 `configs/llm/sft_debug.json` 里把 `sft_jsonl` 改成数组：

```json
"sft_jsonl": [
  "data/text/raw/sft_chat_zh.jsonl",
  "data/text/raw/sft_open_everyday.jsonl"
]
```

SFT 的关键区别：

```text
<|system|> / <|user|> / 历史 <|assistant|>：labels = -100，不参与 loss
最后一条 <|assistant|> 回答正文：正常计算 loss
<|end|>：正常计算 loss，让模型学会停止当前轮回复
```

这样模型不会重点学习“继续伪造用户问题”，而是学习“看到用户问题后生成助手回答”。

产物：

```text
checkpoints/llm/sft_debug/last.pt
```

验收标准：

- SFT train loss 能跑通并下降。
- 生成结果不会明显继续输出 `<|user|>`、`<|system|>` 等角色 token。

## 6. 冒烟评估

```bash
python scripts/llm/evaluate_chat.py --checkpoint checkpoints/llm/sft_debug/last.pt
```

默认会测试几类固定问题，并保存：

```text
runs/llm/chat_eval.jsonl
```

当前检查项很简单：

- 回答不是空的。
- 回答里没有继续出现 `<|user|>`、`<|system|>` 等角色 token。
- 回答没有异常过长。

真实项目会继续加入人工标注评估、事实性评估、安全评估、延迟评估和成本评估。

## 7. 导出部署包

```bash
python scripts/llm/export_model.py --checkpoint checkpoints/llm/sft_debug/last.pt --out deployments/llm/chat_debug
```

产物：

```text
deployments/llm/chat_debug/model.pt
deployments/llm/chat_debug/tokenizer.json
deployments/llm/chat_debug/manifest.json
```

部署服务只依赖这个目录，避免线上代码到处读取训练目录。

## 8. 启动 HTTP 服务

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

返回格式：

```json
{
  "reply": "...",
  "history": "<|system|>...\n<|user|>...\n<|assistant|>..."
}
```

`history` 是内部 ChatML 文本，可以传回下一次请求，用来模拟多轮对话。终端或产品界面可以继续显示成“用户/助手”，但训练和服务内部统一使用 ChatML。

## 9. 一键跑完整流程

如果你想一次串起来：

```bash
python scripts/llm/run_chat_workflow.py
```

如果已经有 base checkpoint，只想重新 SFT、评估和导出：

```bash
python scripts/llm/run_chat_workflow.py --skip-pretrain
```

如果已经有 SFT checkpoint，只想重新评估和导出：

```bash
python scripts/llm/run_chat_workflow.py --skip-pretrain --skip-sft
```

## 10. 真实项目对应关系

| 教学项目阶段 | 真实项目阶段 |
| --- | --- |
| `train_tokenizer.py` | tokenizer 训练或选择 |
| `prepare_data.py` | 数据清洗、切分、打包 |
| `train.py` | base model pretraining |
| `train_sft.py` | supervised fine-tuning |
| `evaluate_chat.py` | offline evaluation / smoke test |
| `export_model.py` | model packaging |
| `serve_chat.py` | online inference service |
| `run_chat_workflow.py` | pipeline / CI job |

## 11. 下一步怎么变得更真实

可以按这个顺序升级：

1. 增加高质量 SFT 数据，不要只靠内置样例。
2. 把评估集单独放到 `data/text/eval/`，不要和训练集混用。
3. 记录每次训练的配置、git commit、数据版本和指标。
4. 加入更严格的停止条件和安全拒答样本。
5. 用更大的中文语料预训练，再用更干净的指令数据 SFT。
6. 部署时加入请求日志、超时、并发限制和回滚机制。

## 12. 常见问题

### checkpoint vocab 和 tokenizer vocab 不匹配

如果 SFT 时报错类似：

```text
checkpoint vocab=645, tokenizer vocab=1024
```

说明这个 checkpoint 不是用当前 `artifacts/llm/tokenizer/tokenizer.json` 训练出来的。

真实项目里 tokenizer 是模型的一部分，不能随便替换。正确做法是：

1. 固定 tokenizer。
2. 用这个 tokenizer 准备预训练数据。
3. 用这个 tokenizer 训练 base checkpoint。
4. 用同一个 tokenizer 做 SFT、评估和部署。

最简单的修复方式是重新跑完整流程：

```bash
python scripts/llm/run_chat_workflow.py
```

如果你只是想复用已有 checkpoint，就必须确认它和当前 tokenizer 是同一套产物。

### 旧 checkpoint 用新 ChatML prompt 输出很乱

如果你把旧 checkpoint 直接拿来跑新 ChatML 模板，可能会看到乱码或格式混乱。

原因是旧 checkpoint 没有用包含 `<|system|>`、`<|user|>`、`<|assistant|>`、`<|end|>` 的 tokenizer 和语料训练。它并不知道这些 token 的聊天含义。

正确做法：

```bash
python scripts/llm/run_chat_workflow.py
```

完整重跑后，新 tokenizer、base checkpoint、SFT checkpoint 和部署包才是一致的。
