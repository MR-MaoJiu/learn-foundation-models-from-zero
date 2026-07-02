from __future__ import annotations

"""
Tokenizer 模块：负责“文字”和“数字 token id”之间的转换。

为什么 LLM 需要 tokenizer？
神经网络只能处理数字，不能直接处理中文字符串。

例如：
    "语言模型" -> [128, 45, 901]

训练完成后生成文本时，还要反过来：
    [128, 45, 901] -> "语言模型"

这个项目使用 byte-level BPE tokenizer。
你暂时可以这样理解 BPE：
1. 一开始把文本拆得很细。
2. 统计哪些片段经常一起出现。
3. 把高频片段合并成一个 token。
4. 重复很多次，得到一个词表。
"""

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


# 特殊 token。
#
# <pad>：补齐用，本项目暂时很少用。
# <unk>：遇到未知 token 时使用。
# <bos>：begin of sequence，表示一段文本开始。
# <eos>：end of sequence，表示一段文本结束。
#
# 下面 4 个 ChatML 风格 token 用来标记聊天角色和单轮结束。
# 主流聊天模型通常不会只靠“用户：/助手：”这类普通文字判断角色，
# 而是会使用清晰、稳定、不容易和正文混淆的特殊边界。
#
# <|system|>：系统指令，规定助手整体行为。
# <|user|>：用户输入。
# <|assistant|>：助手回复开始。
# <|end|>：一条消息结束。生成时遇到它就可以停止当前轮回复。
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def train_bpe_tokenizer(
    input_files: Iterable[str | Path],
    out_dir: str | Path,
    vocab_size: int,
    min_frequency: int = 2,
) -> Path:
    """训练 byte-level BPE tokenizer，并保存 `tokenizer.json`。

    参数：
    - input_files：训练 tokenizer 的文本文件。
    - out_dir：输出目录。
    - vocab_size：词表最大大小。
    - min_frequency：片段至少出现多少次才允许被合并进词表。

    注意：
    tokenizer 的词表不是越大越好。
    - 词表太小：一句话会被切成很多 token，序列变长。
    - 词表太大：输出层变大，模型参数和训练成本增加。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 创建一个 BPE tokenizer。
    # unk_token="<unk>" 表示遇到不认识的片段时，用 <unk> 代替。
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # ByteLevel 可以处理中文、英文、符号、空格等混合文本。
    # 它的一个好处是可逆性比较强，decode 后能尽量还原原文。
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # BpeTrainer 负责真正学习词表。
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # tokenizers 库需要字符串路径。
    files = [str(Path(p)) for p in input_files]

    # 读取语料并学习 BPE 合并规则。
    tokenizer.train(files, trainer=trainer)

    # 保存成一个 JSON 文件，后续训练和生成都会加载它。
    output_path = out_dir / "tokenizer.json"
    tokenizer.save(str(output_path))
    return output_path


class LLMTokenizer:
    """对 `tokenizers.Tokenizer` 做一个小封装。

    这样项目其它地方不用直接接触第三方库细节，只需要调用：
    - encode：文字 -> token id
    - decode：token id -> 文字
    - vocab_size：词表大小
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        # 从 tokenizer.json 加载 tokenizer。
        self.tokenizer = Tokenizer.from_file(str(self.path))

        # 记录特殊 token 的 id。
        # 例如 <eos> 在词表里可能是 3。
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.chat_system_id = self.tokenizer.token_to_id("<|system|>")
        self.chat_user_id = self.tokenizer.token_to_id("<|user|>")
        self.chat_assistant_id = self.tokenizer.token_to_id("<|assistant|>")
        self.chat_end_id = self.tokenizer.token_to_id("<|end|>")

    @property
    def vocab_size(self) -> int:
        """返回 tokenizer 的真实词表大小。

        为什么要用真实大小？
        因为小语料可能训练不出你请求的完整 vocab_size。
        例如你请求 8000，但语料太小，实际可能只有几百个 token。
        模型必须和 tokenizer 的真实词表大小一致。
        """

        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """把文本转换成 token id 列表。

        add_bos：
        - 是否在开头加 <bos>。
        - 生成文本时常用，表示“从这里开始”。

        add_eos：
        - 是否在结尾加 <eos>。
        - 准备训练语料时常用，表示“一段文档结束”。
        """

        ids = self.tokenizer.encode(text).ids

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """把 token id 列表转换回文本。

        普通展示时通常跳过特殊 token。
        聊天推理内部会保留特殊 token，这样才能看到 `<|end|>` 并正确截断。
        """

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
