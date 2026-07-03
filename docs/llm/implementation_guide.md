# 代码方法导读

这份文档按文件解释每个核心方法的作用。建议你一边打开源码一边看。

## `src/foundation_models/llm/config.py`

`ModelConfig`

保存模型结构参数，例如词表大小、层数、注意力头数、隐藏维度、上下文长度。模型初始化时只需要传入这个对象，就能知道自己该建多大。

`ModelConfig.__post_init__()`

创建配置后自动检查参数是否合法。这里检查 `n_embd` 是否能被 `n_head` 整除，因为每个注意力头都要分到同样大小的向量。

`ModelConfig.head_dim`

计算每个注意力头的维度。比如隐藏维度是 768，头数是 12，则每个头处理 64 维。

`load_json_config(path)`

读取 JSON 配置文件，返回普通 Python 字典。

`model_config_from_dict(values)`

把字典转换成 `ModelConfig`。它会检查有没有拼错的配置项。

## `src/foundation_models/llm/tokenizer.py`

`train_bpe_tokenizer(input_files, out_dir, vocab_size)`

训练 byte-level BPE tokenizer。它会阅读原始 `.txt` 文件，学习常见文字片段，然后保存 `tokenizer.json`。

`LLMTokenizer.__init__(path)`

加载 tokenizer 文件，并记录 `<pad>`、`<unk>`、`<bos>`、`<eos>` 这些特殊 token 的 id。

`LLMTokenizer.vocab_size`

返回 tokenizer 的真实词表大小。训练脚本会用它覆盖配置里的 `vocab_size`，避免模型词表和 tokenizer 不一致。

`LLMTokenizer.encode(text, add_bos=False, add_eos=False)`

把字符串变成 token id 列表。`add_bos` 在开头加开始标记，`add_eos` 在结尾加结束标记。

`LLMTokenizer.decode(ids)`

把 token id 列表还原成文本。

聊天相关特殊 token：

- `<|system|>`：系统指令开始。
- `<|user|>`：用户消息开始。
- `<|assistant|>`：助手回复开始。
- `<|end|>`：一条消息结束。

这些 token 会被注册进 tokenizer，避免模型把角色边界当成普通正文片段。

## `src/foundation_models/llm/data.py`

`read_text_files(paths)`

读取多个文本文件，并按空行切成多个小文档。

`tokenize_documents(tokenizer, documents)`

把文档列表转换成一条很长的 token 流。每个文档末尾会加 `<eos>`。

`save_token_array(ids, path, vocab_size)`

把 token id 保存成 `.bin` 二进制文件。词表小于 65536 时使用 `uint16`，可以节省硬盘空间。

`make_train_val_bins(...)`

从原始文本生成 `train.bin` 和 `val.bin`。训练集用于更新参数，验证集用于观察模型是否过拟合。

`BinaryTokenDataset.__init__(path, block_size, vocab_size)`

用 `np.memmap` 打开 `.bin` 文件。memmap 不会一次性把大文件全部加载进内存，适合训练大语料。

`BinaryTokenDataset.get_batch(batch_size, device)`

随机抽取一批训练样本。返回 `x` 和 `y`，其中 `y` 是 `x` 向右移动一个 token 的答案。

## `src/foundation_models/llm/chat.py`

`render_chat_messages(messages, add_generation_prompt)`

把 OpenAI/GPT 风格的 `messages` 渲染成 ChatML 文本。SFT、生成、评估和部署都复用这套模板，避免训练和推理格式不一致。

`build_chat_prompt(prompt)`

把普通用户输入包装成 system + user + assistant generation prompt。

`generate_chat_reply(...)`

生成一轮助手回复。它会在生成时使用 `<|end|>` 和 `<eos>` 作为停止 token，减少模型继续生成下一轮用户的情况。

## `src/foundation_models/llm/sft_data.py`

`read_sft_jsonl(path)`

读取 SFT JSONL。推荐格式是：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

`validate_messages(messages)`

检查 `system/user/assistant` 顺序是否适合 SFT：system 只能第一条，user 和 assistant 要交替，最后一条必须是 assistant。

`ChatSFTDataset.encode_example(example)`

把一条 SFT 样本转成 `input_ids` 和 `labels`。prompt 部分的 labels 会设置成 `-100`，不参与 loss；最后 assistant 回答和 `<|end|>` 会参与 loss。

## `src/foundation_models/llm/model.py`

`RMSNorm`

归一化层，让每层输入的数值更稳定。

`rotate_half(x)`

RoPE 的辅助函数，把 `[a, b]` 变成 `[-b, a]`，用于构造旋转。

`build_rope_cache(seq_len, head_dim, theta, device)`

提前计算 RoPE 需要的 `cos` 和 `sin`。这样每层注意力都能复用位置编码。

`apply_rope(x, cos, sin)`

把 RoPE 应用到 query 和 key。这样注意力计算就能感知 token 的位置。

`CausalSelfAttention`

因果自注意力层。它让每个 token 只能看见自己和左边的 token，不能偷看未来。

`SwiGLU`

前馈网络层。它在每个 token 位置上独立工作，负责增强非线性表达能力。

`TransformerBlock`

一个完整 Transformer 层：RMSNorm、Attention、残差连接、RMSNorm、MLP、残差连接。

`GPT.forward(input_ids, labels=None)`

模型前向传播。输入 token id，输出 logits。如果传入 labels，就同时计算下一个 token 预测的交叉熵 loss。

`GPT.generate(...)`

生成文本。每次预测一个新 token，把它拼到输入后面，再继续预测。

`GPT.num_parameters()`

统计模型中需要训练的参数数量。

## `scripts/llm/train.py`

`estimate_loss(...)`

在验证集上跑几个 batch，计算平均 loss。

`main()`

训练入口。它负责读取配置、加载 tokenizer、创建数据集、创建模型、设置优化器、执行训练循环、定期验证和保存 checkpoint。

训练循环里的关键步骤是：

1. 取一批 `x, y`。
2. 模型预测 logits 和 loss。
3. `loss.backward()` 计算梯度。
4. 梯度裁剪，避免梯度爆炸。
5. `optimizer.step()` 更新参数。
6. 按间隔打印日志、验证、保存模型。

## `scripts/llm/download_open_chinese_corpus.py`

这个脚本是可选的，用来下载一小份开源中文语料。

它默认使用 Hugging Face 的 `wikimedia/wikipedia` 数据集中文子集 `20231101.zh`，并通过 streaming 流式读取，避免一次性下载整个数据集。

主要参数：

`--out`

输出文本文件路径，默认是 `data/text/raw/open_zh_wikipedia.txt`。

`--max-docs`

最多写入多少篇文章。默认 200，适合先学习流程。

`--min-chars`

跳过太短的文章，默认少于 200 字符的文章不写入。

`--max-chars`

限制总字符数。默认 0 表示不限制。

脚本会额外生成一个 `.source.md` 文件，用来记录数据来源、子集、写入数量和许可提醒。

## `scripts/llm/build_pretrain_corpus.py`

这个脚本用来构建默认预训练文本 `data/text/raw/pretrain_zh.txt`。

它会组合三类内容：

1. `tiny_zh_corpus.txt` 里的原创种子文本。
2. `sft_chat_zh.jsonl` 转写出的可读对话文本。
3. 合成的中文技术和日常段落。

这样做是为了避免 tiny 语料太少，导致 `val.bin` token 数小于 `max_seq_len`。它仍然是教学语料，不是高质量真实预训练数据的替代品。

## `scripts/llm/build_sft_chat_corpus.py`

这个脚本生成中文 `messages` JSONL，用于 SFT。

新版生成器会高频覆盖天气、语言模型、tokenizer、部署检查、模型续写用户、CUDA、NaN、睡眠、写作和任务规划等常见问题。每条样本都保持 `system/user/assistant` 结构，避免模型学习旧式“用户：/助手：”全文续写。
