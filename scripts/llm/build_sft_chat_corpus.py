from __future__ import annotations

"""
Generate Chinese chat SFT data in OpenAI/GPT-style `messages` JSONL.

The file is used for supervised fine-tuning. Each row looks like:

    {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]}

The generator intentionally avoids old free-form transcripts like
"用户：... 助手：...". Structured messages let the training code mask the prompt
and supervise only the assistant answer.
"""

import argparse
import json
from pathlib import Path
import random


SYSTEM_GENERAL = "你是一个中文 AI 助手。回答要准确、自然、简洁；不知道实时信息时要说明限制。"
SYSTEM_TEACHER = "你是一个中文 AI 助手。用适合初学者的方式解释概念，必要时给一个小例子。"
SYSTEM_ENGINEER = "你是一个中文 AI 助手。回答代码和工程问题时先定位原因，再给可执行步骤。"
SYSTEM_WRITER = "你是一个中文 AI 助手。帮助用户改写文本时保留原意，让表达更清楚自然。"
SYSTEM_PLANNER = "你是一个中文 AI 助手。帮助用户拆解任务，给出清晰、可执行的下一步。"
SYSTEM_SAFETY = "你是一个中文 AI 助手。遇到实时、医疗、法律、金融或隐私问题时说明限制，并建议用户核实可靠来源。"


def msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content.strip()}


def single_turn(system: str, user: str, assistant: str) -> dict[str, list[dict[str, str]]]:
    return {"messages": [msg("system", system), msg("user", user), msg("assistant", assistant)]}


def multi_turn(system: str, turns: list[tuple[str, str]]) -> dict[str, list[dict[str, str]]]:
    messages = [msg("system", system)]
    for user_text, assistant_text in turns:
        messages.append(msg("user", user_text))
        messages.append(msg("assistant", assistant_text))
    return {"messages": messages}


CORE_QA = [
    {
        "system": SYSTEM_SAFETY,
        "questions": [
            "今天天气怎么样？",
            "今天外面适合出门吗？",
            "你知道今天上海下雨吗？",
            "帮我看看今天要不要带伞。",
            "今天冷不冷？",
        ],
        "answer": "我不能直接获取实时天气。你可以查看天气应用里的温度、降雨概率和风力；如果不确定，出门带伞和薄外套会更稳妥。",
    },
    {
        "system": SYSTEM_TEACHER,
        "questions": [
            "语言模型是什么？",
            "你能解释一下语言模型是什么吗？",
            "LLM 到底在学什么？",
            "大语言模型为什么能生成文字？",
            "语言模型和聊天机器人有什么关系？",
        ],
        "answer": "语言模型的核心任务是根据前面的 token 预测下一个 token。预训练让它学会续写文本，SFT 再让它学会按照用户问题生成助手回复。",
    },
    {
        "system": SYSTEM_ENGINEER,
        "questions": [
            "为什么我的模型会一直续写用户和助手？",
            "生成结果总是继续写下一轮用户怎么办？",
            "小模型为什么会输出用户：助手：这种格式？",
            "SFT 后模型还在扮演双方对话，怎么排查？",
            "模型不会停在助手回答处怎么办？",
        ],
        "answer": "通常是模板和监督目标不一致。请确认训练数据是 messages 格式，SFT 时只监督 assistant 回复部分，并在生成时把 `<|end|>` 作为停止 token。",
    },
    {
        "system": SYSTEM_TEACHER,
        "questions": [
            "训练 tokenizer 的作用是什么？",
            "为什么训练前要先有 tokenizer？",
            "tokenizer 是干什么的？",
            "为什么 checkpoint 不能随便换 tokenizer？",
            "词表大小会影响什么？",
        ],
        "answer": "tokenizer 负责把文本和 token id 互相转换。模型的输入输出层都依赖同一套词表，所以 tokenizer、预训练 checkpoint、SFT checkpoint 和部署包必须匹配。",
    },
    {
        "system": SYSTEM_PLANNER,
        "questions": [
            "部署前需要检查什么？",
            "模型上线前要做哪些检查？",
            "导出模型前应该确认什么？",
            "本地聊天服务启动前要检查什么？",
            "怎么确认训练到部署是同一套产物？",
        ],
        "answer": "部署前至少检查 tokenizer 和 checkpoint 是否匹配、评估是否通过、生成是否能停止、部署包是否包含 model、tokenizer 和 manifest，并用 `/health` 和 `/chat` 做一次接口测试。",
    },
    {
        "system": SYSTEM_ENGINEER,
        "questions": [
            "CUDA 不可用怎么办？",
            "torch.cuda.is_available() 是 False 怎么办？",
            "训练没用上显卡怎么排查？",
            "PyTorch 识别不到 GPU 怎么办？",
        ],
        "answer": "先运行 `python -c \"import torch; print(torch.cuda.is_available())\"`。如果是 False，检查 PyTorch 是否安装了 CUDA 版本、显卡驱动是否正常，以及当前虚拟环境是否装对依赖。",
    },
    {
        "system": SYSTEM_ENGINEER,
        "questions": [
            "训练 loss 变成 NaN 怎么办？",
            "模型训练突然不稳定怎么办？",
            "loss 爆炸应该怎么排查？",
            "为什么训练到一半 loss 是 nan？",
        ],
        "answer": "常见原因是学习率太大、梯度爆炸或混合精度不稳定。可以先降低学习率，开启梯度裁剪，并用 fp32 小规模跑通来定位问题。",
    },
    {
        "system": SYSTEM_TEACHER,
        "questions": [
            "SFT 是什么？",
            "监督微调有什么用？",
            "预训练之后为什么还要 SFT？",
            "SFT 和预训练有什么区别？",
        ],
        "answer": "SFT 是监督微调，用结构化问答样本继续训练模型。预训练主要学语言分布，SFT 主要学如何按指令回答，以及什么时候停止当前轮回复。",
    },
    {
        "system": SYSTEM_GENERAL,
        "questions": [
            "最近晚上睡不着怎么办？",
            "我睡眠不太好，有什么建议？",
            "晚上总是刷手机停不下来怎么办？",
            "想改善睡眠应该从哪里开始？",
        ],
        "answer": "可以先固定睡前流程：提前半小时放下手机、调暗灯光、避免咖啡因。如果持续影响生活或明显加重，建议咨询医生。",
    },
    {
        "system": SYSTEM_WRITER,
        "questions": [
            "帮我写个通知，说今晚系统维护。",
            "帮我把系统维护通知写得正式一点。",
            "今晚系统维护，怎么通知用户？",
            "写一句简短维护公告。",
        ],
        "answer": "通知：今晚系统将进行维护，期间部分功能可能短暂不可用。维护完成后会尽快恢复正常使用，感谢理解。",
    },
    {
        "system": SYSTEM_PLANNER,
        "questions": [
            "帮我规划一下训练一个小 LLM。",
            "我想从零做一个小语言模型，怎么开始？",
            "训练小模型的步骤是什么？",
            "从数据到部署怎么安排？",
        ],
        "answer": "可以按这几步走：准备语料，训练 tokenizer，准备 train.bin 和 val.bin，预训练 base model，用 messages 数据做 SFT，最后评估、导出并启动服务。",
    },
]


GENERAL_QA = [
    ("为什么要做验证集？", "验证集不参与参数更新，用来观察模型对未见数据的表现。如果训练 loss 降而验证 loss 升，可能已经过拟合。", SYSTEM_TEACHER),
    ("如何检查模型是否过拟合？", "看训练 loss 和验证 loss 的走势。如果训练集越来越好、验证集变差，通常说明模型在记忆训练数据。", SYSTEM_TEACHER),
    ("为什么要固定随机种子？", "固定随机种子能减少实验波动，方便复现问题。它不能保证所有设备完全一致，但能让调试更稳定。", SYSTEM_ENGINEER),
    ("如何设计 SFT 数据格式？", "建议使用 messages JSONL，明确 system、user、assistant。最后一条必须是 assistant，训练时只监督这部分。", SYSTEM_ENGINEER),
    ("怎样让模型回答更自然？", "需要高质量、多样化的 SFT 样本，也要让训练和推理使用同一套聊天模板。小模型还需要控制生成温度和停止条件。", SYSTEM_TEACHER),
    ("如何记录一次训练实验？", "记录数据版本、配置文件、随机种子、代码提交、训练日志、评估结果和 checkpoint 路径。这样后续才能复现。", SYSTEM_PLANNER),
    ("如何排查生成乱码？", "先检查语料编码是否是 UTF-8，再确认 tokenizer 和 checkpoint 匹配，最后检查 decode 时是否跳过了不该跳过的特殊 token。", SYSTEM_ENGINEER),
    ("batch size 怎么选？", "先根据显存选能稳定运行的 micro batch，再用梯度累积得到目标 batch size。显存不足时优先减小 micro batch。", SYSTEM_ENGINEER),
    ("学习率怎么理解？", "学习率决定每次参数更新走多大一步。太大容易不稳定，太小训练很慢；通常需要 warmup 和逐步衰减。", SYSTEM_TEACHER),
    ("怎么减少模型胡编？", "要增加可靠数据和评估，降低生成温度，并让模型在不知道实时或高风险信息时明确说明限制。", SYSTEM_SAFETY),
    ("README 应该写什么？", "至少写清项目目标、安装方式、最短运行命令、分步流程、数据格式、产物目录和常见问题。", SYSTEM_WRITER),
    ("如何做一次技术分享？", "先确定听众和目标，再列 3 到 5 个核心点，准备一个可运行示例，最后预留答疑时间。", SYSTEM_PLANNER),
    ("我想买耳机怎么选？", "先确定预算和用途。通勤多看降噪和佩戴舒适度，游戏多看延迟和麦克风，运动多看稳固和防水。", SYSTEM_GENERAL),
    ("周末短途旅行怎么准备？", "先确认交通、住宿和天气，再准备证件、充电器、常用药和一套备用衣物。行李不用太多。", SYSTEM_GENERAL),
    ("怎么和家人商量家务分工？", "先说共同目标，再提出具体请求。例如：我希望周末大家都轻松一点，我们能不能把家务分一下。", SYSTEM_GENERAL),
]


QUESTION_PREFIXES = ["", "请简短回答：", "帮我解释一下：", "我不太懂，", "用初学者能懂的话说，", "请按步骤说："]
ANSWER_PREFIXES = ["", "简短说，", "可以这样理解：", "先给结论："]
FOLLOW_UPS = [
    ("能再简单一点吗？", "可以。先记住一个核心点：训练和部署要使用同一套数据格式、tokenizer 和 checkpoint，任何一环混用都容易出错。"),
    ("我下一步该做什么？", "下一步先跑最小闭环：生成语料、校验格式、训练 tokenizer、准备 `.bin`，再启动训练。"),
    ("常见坑是什么？", "常见坑是语料太少、编码乱码、tokenizer 和 checkpoint 不匹配、SFT 没有 mask prompt、生成时缺少停止 token。"),
]


def add_variants(
    examples: list[dict[str, list[dict[str, str]]]],
    system: str,
    questions: list[str],
    answer: str,
    repeats: int,
    rng: random.Random,
) -> None:
    for index in range(repeats):
        question = questions[index % len(questions)]
        prefix = QUESTION_PREFIXES[(index // len(questions)) % len(QUESTION_PREFIXES)]
        answer_prefix = ANSWER_PREFIXES[index % len(ANSWER_PREFIXES)]
        user_text = f"{prefix}{question}" if prefix else question
        assistant_text = f"{answer_prefix}{answer}" if answer_prefix else answer
        examples.append(single_turn(system, user_text, assistant_text))

        if index % 7 == 0:
            follow_user, follow_answer = rng.choice(FOLLOW_UPS)
            examples.append(
                multi_turn(
                    system,
                    [
                        (user_text, assistant_text),
                        (follow_user, follow_answer),
                    ],
                )
            )


def build_examples(target_count: int, seed: int) -> list[dict[str, list[dict[str, str]]]]:
    rng = random.Random(seed)
    examples: list[dict[str, list[dict[str, str]]]] = []

    # Oversample core prompts so a tiny model sees the important chat behaviors
    # many times during short debug SFT runs.
    for item in CORE_QA:
        add_variants(
            examples=examples,
            system=item["system"],
            questions=item["questions"],
            answer=item["answer"],
            repeats=160,
            rng=rng,
        )

    for question, answer, system in GENERAL_QA:
        add_variants(
            examples=examples,
            system=system,
            questions=[question],
            answer=answer,
            repeats=60,
            rng=rng,
        )

    contexts = [
        "我正在从零学习 LLM，",
        "我想写到项目文档里，",
        "我准备给同学讲一遍，",
        "我想先跑通最小实验，",
        "我担心部署时出错，",
        "请用自然聊天的方式回答，",
        "请给我一个可执行建议，",
        "请不要太长，",
    ]

    flat_bank: list[tuple[str, str, str]] = []
    for item in CORE_QA:
        for question in item["questions"]:
            flat_bank.append((question, item["answer"], item["system"]))
    flat_bank.extend(GENERAL_QA)

    while len(examples) < target_count:
        question, answer, system = rng.choice(flat_bank)
        user_text = f"{rng.choice(contexts)}{question}"
        if rng.random() < 0.35:
            user_text += " 请给一个例子。"
        if rng.random() < 0.25:
            user_text += " 请分步骤回答。"

        assistant_text = answer
        if "分步骤" in user_text:
            assistant_text = f"可以分三步看：1. 先确认目标和输入。2. 再检查数据、tokenizer 和 checkpoint 是否一致。3. 最后用评估和实际请求验证结果。{answer}"
        elif "例子" in user_text:
            assistant_text = f"{answer} 举例来说，如果你重新训练 tokenizer，就需要用它重新准备数据、训练 checkpoint，并用同一套产物部署。"

        examples.append(single_turn(system, user_text, assistant_text))

    # Stable de-duplication, then fill again if de-dup removed too much.
    seen: set[str] = set()
    unique: list[dict[str, list[dict[str, str]]]] = []
    for item in examples:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)

    suffix = 0
    while len(unique) < target_count:
        question, answer, system = rng.choice(flat_bank)
        suffix += 1
        item = single_turn(system, f"场景 {suffix}：{question}", answer)
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)

    rng.shuffle(unique)
    return unique[:target_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Chinese chat SFT corpus in messages JSONL format.")
    parser.add_argument("--out", default="data/text/raw/sft_chat_zh.jsonl")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_examples(target_count=args.count, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote {len(examples)} SFT examples to {out_path}")


if __name__ == "__main__":
    main()
