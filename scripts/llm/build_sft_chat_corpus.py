from __future__ import annotations

"""
构建中文聊天 SFT 语料。

这个脚本生成标准 OpenAI/GPT `messages` JSONL：

    {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]}

为什么不用旧的“用户：/助手：”普通文本？

- 主流聊天模型通常使用结构化 messages，再渲染成 ChatML 或类似模板。
- 结构化数据能明确区分 system/user/assistant，不容易训练出“继续写下一轮用户”的坏习惯。
- SFT 时可以只监督最后一条 assistant 回答，和真实聊天部署目标一致。

这份生成器不是为了替代真实高质量人工数据，而是为了让教学项目有足够覆盖面的
可复现语料：日常沟通、学习解释、代码帮助、计划安排、写作润色、信息限制、
拒答边界、情绪支持、多轮上下文等。
"""

import argparse
import json
from pathlib import Path
import random
from typing import Iterable


SYSTEM_GENERAL = "你是一个中文 AI 助手。回答要准确、自然、简洁；不知道实时信息时要说明限制。"
SYSTEM_TEACHER = "你是一个中文 AI 助手。用适合初学者的方式解释概念，必要时给一个小例子。"
SYSTEM_ENGINEER = "你是一个中文 AI 助手。回答代码和工程问题时先定位原因，再给可执行步骤。"
SYSTEM_WRITER = "你是一个中文 AI 助手。帮助用户改写文本时保留原意，让表达更清楚自然。"
SYSTEM_PLANNER = "你是一个中文 AI 助手。帮助用户拆解任务，给出清晰、可执行的下一步。"
SYSTEM_SAFETY = "你是一个中文 AI 助手。遇到实时、医疗、法律、金融或隐私问题时说明限制，并建议用户核实可靠来源。"


def row(messages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {"messages": messages}


def msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def single_turn(system: str, user: str, assistant: str) -> dict[str, list[dict[str, str]]]:
    return row([msg("system", system), msg("user", user), msg("assistant", assistant)])


def multi_turn(system: str, turns: list[tuple[str, str]]) -> dict[str, list[dict[str, str]]]:
    messages = [msg("system", system)]
    for user_text, assistant_text in turns:
        messages.append(msg("user", user_text))
        messages.append(msg("assistant", assistant_text))
    return row(messages)


def seed_examples() -> list[dict[str, list[dict[str, str]]]]:
    """少量手写高质量样例，给整体语气定调。"""

    return [
        single_turn(
            SYSTEM_GENERAL,
            "今天天气怎么样？",
            "我不能直接看到实时天气。你可以告诉我所在城市，或者查看天气应用里的温度、降雨概率和风力；如果不确定，出门带伞和薄外套会更稳妥。",
        ),
        single_turn(
            SYSTEM_TEACHER,
            "你能帮我解释语言模型是什么吗？",
            "可以。语言模型会根据前面已经出现的 token 预测下一个 token。训练时它不断比较预测和真实答案的差距，再通过反向传播调整参数。",
        ),
        single_turn(
            SYSTEM_TEACHER,
            "从零训练和 SFT 有什么区别？",
            "结论：从零训练学语言基础，SFT 学如何按指令回答。从零训练通常从随机参数开始，需要大量语料和算力；SFT 是在已有模型上继续用问答数据训练，让模型更像聊天助手。",
        ),
        single_turn(
            SYSTEM_ENGINEER,
            "为什么我的小模型会一直续写用户和助手？",
            "因为它学到的是继续补全训练文本，而不是只回答当前问题。需要在推理时使用聊天模板和停止条件，并在 SFT 时只监督助手回答部分。",
        ),
        multi_turn(
            SYSTEM_PLANNER,
            [
                ("我准备做一个小语言模型。", "可以，建议先把 tokenizer、数据处理、训练循环和生成脚本跑通。"),
                ("那下一步应该先做什么？", "下一步先准备一份小而干净的语料，训练 tokenizer，并确认同一个 tokenizer 会贯穿预训练、SFT、评估和部署。"),
            ],
        ),
    ]


def daily_examples() -> Iterable[dict[str, list[dict[str, str]]]]:
    topics = [
        ("买菜", "家里还缺鸡蛋和青菜。", "可以先列一个短清单：鸡蛋、青菜、豆腐。到店后再看水果是否新鲜。"),
        ("做饭", "我今晚想做饭但时间不多。", "可以选一道主菜加一个快手蔬菜，比如番茄鸡蛋配清炒青菜，步骤少也比较稳。"),
        ("通勤", "明早我怕上班迟到。", "建议提前十到十五分钟出门，并准备一个备选路线，比如地铁延误时改坐公交或打车。"),
        ("运动", "我很久没锻炼了，怎么开始？", "先从低强度开始，比如每天快走十五分钟。连续一周后，再慢慢增加时长或强度。"),
        ("睡眠", "最近晚上很难睡着。", "可以先固定睡前流程：提前半小时放下手机、调暗灯光、避免咖啡因。如果持续影响生活，建议咨询医生。"),
        ("购物", "我想买耳机但不知道怎么选。", "先确定预算和主要用途。如果通勤多，优先看降噪和佩戴舒适度；如果打游戏多，关注延迟和麦克风。"),
        ("旅行", "周末短途旅行怎么准备？", "先确定交通、住宿和天气，再准备证件、充电器、常用药和一套备用衣物。行李不用太多。"),
        ("家庭沟通", "我想和家人商量分工，但怕吵起来。", "可以先说共同目标，再说具体请求，例如“我希望周末大家都轻松一点，我们能不能把家务分一下”。"),
    ]
    for topic, user, assistant in topics:
        yield single_turn(SYSTEM_GENERAL, user, assistant)
        yield multi_turn(
            SYSTEM_GENERAL,
            [
                (f"我在{topic}这件事上有点拿不定主意。", "可以，我们先把目标、限制和最担心的问题说清楚。"),
                ("我想要简单一点，不要太麻烦。", "那就先选最能解决主要问题的一步，其他优化可以后面再做。"),
                ("你能帮我总结一下吗？", "可以：先明确目标，再做最关键的一步，最后根据结果微调。"),
            ],
        )


def learning_examples() -> Iterable[dict[str, list[dict[str, str]]]]:
    concepts = [
        ("tokenizer", "把文字和 token id 互相转换。模型只能处理数字，所以文本进入模型前要先经过 tokenizer。"),
        ("embedding", "把 token id 映射成向量。训练后，相似含义或相似用法的 token 往往会在向量空间中更接近。"),
        ("attention", "让当前位置根据上下文选择要关注的信息。它能帮助模型处理指代、依赖关系和长距离上下文。"),
        ("loss", "衡量模型预测和真实答案之间的差距。训练目标就是让 loss 在训练集和验证集上都尽量降低。"),
        ("SFT", "监督微调，用问答样本训练模型按指令回答。通常只让 assistant 回答部分参与 loss。"),
        ("checkpoint", "训练过程保存下来的模型参数和配置。它可以用于继续训练、评估、生成和部署。"),
        ("temperature", "控制生成随机性。值越低越稳定，值越高越发散。"),
        ("top_k", "限制每次只从概率最高的 k 个 token 中采样，可以减少离谱输出。"),
    ]
    for concept, explanation in concepts:
        yield single_turn(SYSTEM_TEACHER, f"请解释一下 {concept}。", explanation)
        yield single_turn(SYSTEM_TEACHER, f"{concept} 有什么用？", explanation)
        yield multi_turn(
            SYSTEM_TEACHER,
            [
                (f"我不太理解 {concept}。", explanation),
                ("能不能再简单一点？", f"可以。你可以先把 {concept} 理解成模型流程里的一个固定工具，用来让训练或生成更稳定、更可控。"),
            ],
        )


def coding_examples() -> Iterable[dict[str, list[dict[str, str]]]]:
    issues = [
        ("Python 脚本找不到模块", "先确认运行目录是否是项目根目录，再检查是否把 `src` 加入了 Python 搜索路径，最后确认虚拟环境依赖已安装。"),
        ("CUDA 不可用", "先运行 `python -c \"import torch; print(torch.cuda.is_available())\"`。如果是 False，检查 PyTorch 是否安装了 CUDA 版本，以及显卡驱动是否正常。"),
        ("训练 loss 变成 NaN", "常见原因是学习率太大、梯度爆炸或混合精度不稳定。可以降低学习率、开启梯度裁剪，并先用 fp32 小规模验证。"),
        ("checkpoint 和 tokenizer 词表不匹配", "说明它们不是同一套训练产物。应该固定 tokenizer 后重新准备数据并训练 checkpoint，或换成匹配的 checkpoint。"),
        ("生成结果一直重复", "可以降低 temperature，调整 top_k，检查训练数据是否重复，并确认生成时有停止 token。"),
        ("SFT 后仍然续写用户", "检查 labels mask 是否只监督 assistant，确认模板里有明确角色 token，并让模型学习 `<|end|>`。"),
    ]
    for issue, fix in issues:
        yield single_turn(SYSTEM_ENGINEER, f"我遇到问题：{issue}，怎么办？", fix)
        yield multi_turn(
            SYSTEM_ENGINEER,
            [
                (f"{issue}。", "先别急，我们先定位最可能的原因。"),
                ("应该从哪里开始查？", fix),
            ],
        )


def writing_examples() -> Iterable[dict[str, list[dict[str, str]]]]:
    pairs = [
        ("这个模型现在不太行，回答很怪。", "这个模型目前还不稳定，生成结果有时会偏离预期。"),
        ("帮我写个通知，说今晚系统维护。", "通知：今晚系统将进行维护，期间部分功能可能短暂不可用。维护完成后会尽快恢复正常使用。"),
        ("我想表达这个任务很重要但别太严肃。", "可以写成：这个任务对后续进度很关键，我们尽量认真推进，同时保持节奏稳定。"),
        ("帮我把这句话变礼貌：你赶紧把文件发我。", "可以改成：方便的话，请尽快把文件发给我，我这边需要继续处理后续工作。"),
        ("帮我写一句感谢同事帮忙的话。", "谢谢你今天帮忙一起处理这件事，省了我很多时间，也让流程顺利了不少。"),
    ]
    for user, assistant in pairs:
        yield single_turn(SYSTEM_WRITER, user, assistant)


def safety_examples() -> Iterable[dict[str, list[dict[str, str]]]]:
    examples = [
        ("给我一个稳赚的投资建议。", "我不能保证任何投资稳赚。你可以先明确风险承受能力、投资期限和资产配置，并咨询有资质的金融专业人士。"),
        ("我这个症状是不是很严重？", "我不能替代医生诊断。如果症状明显、持续加重或伴随危险信号，建议尽快就医。你可以描述症状、持续时间和已有检查，我可以帮你整理就诊信息。"),
        ("帮我猜一个人的身份证号。", "我不能帮助猜测或获取他人的身份信息。如果你需要核验身份，请通过合法、授权的渠道进行。"),
        ("现在某只股票明天会涨吗？", "我不能预测明天股价一定上涨。可以从公司基本面、行业环境、估值和风险因素入手分析，但最终决策要自行承担风险。"),
        ("你知道今天上海下雨吗？", "我不能直接获取实时天气。建议查看天气应用或气象台信息；如果你告诉我温度、降雨概率和风力，我可以帮你判断出门准备。"),
    ]
    for user, assistant in examples:
        yield single_turn(SYSTEM_SAFETY, user, assistant)


def planning_examples() -> Iterable[dict[str, list[dict[str, str]]]]:
    tasks = [
        ("准备一次技术分享", ["确定主题和听众", "列出 3 到 5 个核心点", "准备示例和演示", "预留答疑时间"]),
        ("整理房间", ["先清理桌面和地面", "把物品分成保留、丢弃、待处理", "最后擦拭和归位"]),
        ("学习 PyTorch", ["先理解 tensor", "再写一个线性模型", "然后跑通训练循环", "最后看 Transformer 结构"]),
        ("找工作", ["明确目标岗位", "整理简历项目", "准备常见面试题", "每周复盘投递结果"]),
        ("训练一个小 LLM", ["准备语料", "训练 tokenizer", "预训练 base model", "用 messages 数据做 SFT", "评估并部署"]),
    ]
    for task, steps in tasks:
        assistant = "可以拆成这几步：\n" + "\n".join(f"{i + 1}. {step}。" for i, step in enumerate(steps))
        yield single_turn(SYSTEM_PLANNER, f"帮我规划一下：{task}。", assistant)


def generate_examples(target_count: int, seed: int) -> list[dict[str, list[dict[str, str]]]]:
    """生成去重后的 SFT 样本。"""

    rng = random.Random(seed)
    base = []
    for fn in [
        seed_examples,
        daily_examples,
        learning_examples,
        coding_examples,
        writing_examples,
        safety_examples,
        planning_examples,
    ]:
        base.extend(list(fn()))

    # 通过轻量变体扩大覆盖面：语气、长度、任务对象都会变化，但仍保持可控。
    tones = ["简短回答", "分步骤回答", "先给结论再解释", "用初学者能懂的话回答"]
    asks = [
        "请帮我解释一下",
        "我不太懂",
        "能不能给我一个建议",
        "帮我快速判断一下",
        "请帮我整理成步骤",
    ]
    subjects = [
        "训练数据为什么重要",
        "为什么要做验证集",
        "如何检查模型是否过拟合",
        "为什么要固定随机种子",
        "如何设计 SFT 数据格式",
        "怎样让模型回答更自然",
        "部署前为什么要做评估",
        "为什么 tokenizer 不能随便换",
        "如何记录一次实验",
        "如何排查生成乱码",
    ]
    contexts = [
        "我正在从零学习",
        "我准备写到项目文档里",
        "我想用它检查训练流程",
        "我准备给同学讲一遍",
        "我想先跑通最小实验",
        "我担心后面部署会出错",
        "我想把流程做得更像真实项目",
        "我希望回答不要太长",
        "我想知道最容易踩坑的地方",
        "我需要一个可以执行的下一步",
    ]
    endings = [
        "请给我一个例子。",
        "请告诉我检查清单。",
        "请用三句话以内说明。",
        "请按步骤回答。",
        "请说明常见错误。",
        "请给出最小可行做法。",
        "请解释为什么。",
        "请帮我判断优先级。",
        "请给我训练时的注意点。",
        "请用自然聊天的方式回答。",
    ]
    for subject in subjects:
        for index, tone in enumerate(tones):
            user = f"{asks[index % len(asks)]}{subject}，{tone}。"
            assistant = f"可以。{subject}的关键是让训练过程可控、可复现，并且让模型学到你真正希望上线时使用的行为。建议先用小样本跑通，再逐步扩大数据和训练步数。"
            base.append(single_turn(SYSTEM_TEACHER, user, assistant))

    seen: set[str] = set()
    unique: list[dict[str, list[dict[str, str]]]] = []
    for item in base:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)

    max_attempts = target_count * 20
    attempts = 0
    while len(unique) < target_count and attempts < max_attempts:
        attempts += 1
        subject = rng.choice(subjects)
        tone = rng.choice(tones)
        context = rng.choice(contexts)
        ending = rng.choice(endings)
        user = f"{context}，{rng.choice(asks)}：{subject}。请{tone}，{ending}"
        assistant = (
            f"好的。关于“{subject}”，可以先抓住一个主线：目标是什么、输入是什么、输出是什么、如何验证。"
            "如果需要更具体的输出，建议先用小样本验证，再扩大到完整数据。"
            "如果是训练流程，就优先保证数据格式、tokenizer、checkpoint 和评估脚本一致。"
        )
        item = single_turn(rng.choice([SYSTEM_GENERAL, SYSTEM_TEACHER, SYSTEM_ENGINEER]), user, assistant)
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)

    if len(unique) < target_count:
        raise RuntimeError(f"Only generated {len(unique)} unique examples; lower --count or add more templates.")

    rng.shuffle(unique)
    return unique[:target_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a larger Chinese chat SFT corpus in messages JSONL format.")
    parser.add_argument("--out", default="data/text/raw/sft_chat_zh.jsonl")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = generate_examples(target_count=args.count, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote {len(examples)} SFT examples to {out_path}")


if __name__ == "__main__":
    main()
