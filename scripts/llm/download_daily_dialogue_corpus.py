from __future__ import annotations

"""
下载和生成“日常生活对话”训练语料。

这个脚本做两类事情：
1. 默认生成原创简体中文日常生活对话。
2. 可选下载英文开源日常对话数据，用于学习多轮对话结构。

生成中文对话的原因：
    公开、可直接下载、字段清晰、许可明确的简体中文“日常闲聊”数据并不总是稳定可用。
    对从 0 学习 LLM 的项目来说，先提供原创简体中文教学语料，可以帮你训练出更像中文日常对话的文本。

注意：
    这些语料适合学习训练流程，不代表高质量商用数据。
    使用开源数据时，请阅读对应数据集页面的许可说明。
"""

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen


def clean_text(text: str) -> str:
    """轻量清洗文本。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_dataset_stream(dataset: str, config: str, split: str):
    """流式加载 Hugging Face 数据集。"""

    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        print("缺少 datasets 依赖。请先运行：python -m pip install -r requirements.txt", file=sys.stderr)
        raise

    return load_dataset(dataset, config, split=split, streaming=True)


def format_turns(turns: Iterable[str], speaker_a: str = "用户", speaker_b: str = "助手") -> str:
    """把多轮话语格式化成适合 LLM 预训练的纯文本。"""

    lines: list[str] = []
    for index, turn in enumerate(turns):
        turn = clean_text(str(turn))
        if not turn:
            continue
        speaker = speaker_a if index % 2 == 0 else speaker_b
        lines.append(f"{speaker}：{turn}")
    return "\n".join(lines)


def iter_dailydialog(max_docs: int) -> Iterable[str]:
    """读取 DailyDialog 英文日常多轮对话。

    这里不用 datasets.load_dataset。
    原因是 roskoN/dailydialog 当前依赖数据集脚本，而新版 datasets 已经不支持执行脚本。
    Hugging Face Dataset Viewer API 可以直接分页读取标准化后的行，更稳定。
    """

    count = 0
    offset = 0
    page_size = 100

    while count < max_docs:
        params = urlencode(
            {
                "dataset": "roskoN/dailydialog",
                "config": "full",
                "split": "train",
                "offset": offset,
                "length": min(page_size, max_docs - count),
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        with urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))

        rows = payload.get("rows", [])
        if not rows:
            break

        for item in rows:
            row = item.get("row", {})
            utterances = row.get("utterances")
            if isinstance(utterances, list):
                text = format_turns(utterances, speaker_a="A", speaker_b="B")
                if text:
                    yield "来源：DailyDialog\n" + text
                    count += 1
                    if count >= max_docs:
                        break

        offset += len(rows)


def iter_everyday_conversations(max_docs: int) -> Iterable[str]:
    """读取 everyday-conversations 英文用户/助手日常主题对话。"""

    dataset = load_dataset_stream("HuggingFaceTB/everyday-conversations-llama3.1-2k", "default", "train_sft")
    count = 0
    for row in dataset:
        messages = row.get("messages")
        if isinstance(messages, list):
            lines = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role", "")
                content = clean_text(str(message.get("content", "")))
                if not content:
                    continue
                speaker = "用户" if role == "user" else "助手"
                lines.append(f"{speaker}：{content}")
            if lines:
                topic = clean_text(str(row.get("full_topic", "")))
                header = f"来源：everyday-conversations\n主题：{topic}" if topic else "来源：everyday-conversations"
                yield header + "\n" + "\n".join(lines)
                count += 1
        if count >= max_docs:
            break


SCENARIOS: dict[str, list[tuple[str, str]]] = {
    "买菜": [
        ("你今天下班顺路买点菜吗？", "可以，家里还缺什么？"),
        ("青菜、鸡蛋和豆腐都快没了。", "好，我再看看水果有没有新鲜的。"),
        ("如果苹果太贵就买香蕉吧。", "没问题，我到超市给你拍一下价格。"),
        ("晚上想吃清淡一点。", "那我买点冬瓜，回去煮汤。"),
    ],
    "做饭": [
        ("今晚你想吃米饭还是面条？", "米饭吧，昨天已经吃过面了。"),
        ("那我炒个番茄鸡蛋，再做个青菜。", "可以，别放太多油。"),
        ("冰箱里还有一点鸡肉。", "那可以切小块炒一下。"),
        ("你大概几点到家？", "半小时后到，我回来帮你洗菜。"),
    ],
    "天气": [
        ("今天外面风好大。", "是啊，出门最好穿厚一点。"),
        ("下午会下雨吗？", "天气预报说傍晚可能有小雨。"),
        ("那我带把伞。", "对，鞋子也别穿太薄。"),
        ("周末要是晴天就去公园走走。", "好，顺便晒晒太阳。"),
    ],
    "通勤": [
        ("你今天坐地铁还是公交？", "地铁吧，公交可能堵车。"),
        ("早高峰人会不会很多？", "会，不过比开车稳定。"),
        ("那你早点出门。", "嗯，我提前十分钟走。"),
        ("到公司记得发个消息。", "好，到了我告诉你。"),
    ],
    "快递": [
        ("你的快递到了，在驿站。", "好的，我下班去取。"),
        ("取件码我发给你了。", "看到了，谢谢。"),
        ("盒子有点大，你拿得动吗？", "应该可以，不行我叫车。"),
        ("里面是你买的书吗？", "对，还有一个电脑支架。"),
    ],
    "约朋友": [
        ("周末有空一起吃饭吗？", "有空啊，你想吃什么？"),
        ("最近想吃火锅。", "可以，我知道一家味道不错。"),
        ("那我们周六晚上去？", "行，我提前订位。"),
        ("要不要叫上小李？", "可以，人多热闹一点。"),
    ],
    "看病": [
        ("你嗓子还疼吗？", "还有一点，早上咳了几次。"),
        ("要不要去医院看看？", "如果下午还不舒服就去。"),
        ("先多喝温水，别吃辣的。", "嗯，我中午吃清淡点。"),
        ("药别乱吃，听医生的。", "知道，我会注意。"),
    ],
    "租房": [
        ("这个房子离地铁近吗？", "走路大概八分钟。"),
        ("采光怎么样？", "下午阳光比较好，客厅挺亮。"),
        ("房租包含物业费吗？", "不包含，物业费每个月另算。"),
        ("那我们周末去看房吧。", "好，我联系中介约时间。"),
    ],
    "学习": [
        ("你今天准备学什么？", "我想把 Python 的函数再复习一遍。"),
        ("要不要一起做几道练习题？", "可以，边做边理解更快。"),
        ("看不懂的地方先标出来。", "嗯，晚上我们一起讨论。"),
        ("别学太晚，注意休息。", "知道，我学到十点就停。"),
    ],
    "工作沟通": [
        ("这个需求今天能整理好吗？", "可以，我下午发第一版。"),
        ("重点写清楚用户流程。", "明白，我会把边界情况也列出来。"),
        ("如果有不确定的地方先标注。", "好，我不会直接猜。"),
        ("整理完我们再一起过一遍。", "没问题。"),
    ],
    "手机": [
        ("你手机电量还够吗？", "只剩百分之二十了。"),
        ("我这里有充电宝。", "太好了，借我用一下。"),
        ("你是不是后台开太多应用了？", "可能是，我等会儿清理一下。"),
        ("屏幕亮度也可以调低一点。", "好，这样应该能多撑一会儿。"),
    ],
    "运动": [
        ("晚上要不要去散步？", "可以，吃完饭走半小时。"),
        ("最近坐太久了，腰有点酸。", "那更应该活动一下。"),
        ("我们走小区旁边那条路吧。", "好，车少也安静。"),
        ("如果下雨就在家拉伸。", "可以，别完全不动。"),
    ],
    "外卖": [
        ("今天不想做饭，要不要点外卖？", "可以，先看看附近有什么清淡一点的。"),
        ("我想吃米粉，你呢？", "我都行，别太辣就好。"),
        ("配送时间大概要四十分钟。", "那现在下单正好。"),
        ("要不要加一份小菜？", "加一份青菜吧，营养均衡一点。"),
    ],
    "超市购物": [
        ("周末去超市吗？", "去吧，家里的日用品快用完了。"),
        ("洗衣液和纸巾都要买。", "我记到购物清单里。"),
        ("如果有打折的牛奶也买一箱。", "好，我会看一下日期。"),
        ("别忘了带环保袋。", "放心，我已经放包里了。"),
    ],
    "银行办事": [
        ("明天我要去银行办张卡。", "记得带身份证。"),
        ("需要提前预约吗？", "有些网点需要，最好先在手机上看一下。"),
        ("如果人很多怎么办？", "可以取号后在附近等一会儿。"),
        ("办完我顺便把账单查一下。", "好，注意别在公共场合说密码。"),
    ],
    "家务": [
        ("今天要不要一起打扫房间？", "可以，先从客厅开始。"),
        ("我负责拖地，你整理桌子。", "好，分工明确快一点。"),
        ("旧纸箱要不要扔掉？", "能回收的先放到门口。"),
        ("床单也该换了。", "那我等会儿拿去洗。"),
    ],
    "睡眠": [
        ("你昨晚睡得好吗？", "一般，半夜醒了一次。"),
        ("是不是睡前看手机太久了？", "可能是，今天早点放下手机。"),
        ("晚上可以喝点温水。", "好，但不要喝太多。"),
        ("明天还要早起。", "那我们十一点前睡吧。"),
    ],
    "旅行": [
        ("下个月想不想出去玩两天？", "可以，先看预算和时间。"),
        ("我想去海边。", "听起来不错，交通也方便。"),
        ("酒店要提前订吗？", "最好提前订，周末容易涨价。"),
        ("行李不用带太多。", "对，轻装出门更轻松。"),
    ],
    "理发": [
        ("我周末想去理发。", "你要剪短一点吗？"),
        ("稍微修一下就行。", "那可以给理发师看参考照片。"),
        ("附近哪家比较好？", "小区门口那家评价还不错。"),
        ("我先打电话问问要不要预约。", "好，免得去了要等很久。"),
    ],
    "宠物": [
        ("猫粮快没了。", "那今天就下单一袋。"),
        ("要不要顺便买猫砂？", "可以买，反正也快用完了。"),
        ("它最近好像不太爱喝水。", "可以换个流动饮水机试试。"),
        ("晚上记得陪它玩一会儿。", "好，让它多活动一下。"),
    ],
    "邻里": [
        ("楼上最近装修有点吵。", "可以先礼貌沟通一下时间。"),
        ("如果中午还施工怎么办？", "那就联系物业确认规定。"),
        ("我不想把关系弄僵。", "语气温和一点，说明影响就好。"),
        ("希望他们能理解。", "大多数人说清楚都会配合。"),
    ],
    "家庭": [
        ("周末要不要回家吃饭？", "可以，我也想看看爸妈。"),
        ("要不要带点水果？", "带一箱橙子吧，他们喜欢。"),
        ("晚上几点过去合适？", "五点左右吧，还能帮忙做饭。"),
        ("吃完饭陪他们散散步。", "好，别吃完就急着走。"),
    ],
}

OPENINGS = [
    "今天过得怎么样？",
    "你现在方便说话吗？",
    "我有件小事想和你商量。",
    "刚才路上有点堵。",
    "我准备出门了。",
    "我刚想起来一件事。",
    "你帮我一起想想可以吗？",
    "我有点拿不准。",
]

CLOSINGS = [
    "好，那就先这样。",
    "行，我记下了。",
    "谢谢你提醒我。",
    "没问题，等会儿联系。",
    "好的，我们按这个来。",
    "听你这么说我放心多了。",
    "那我先去准备。",
    "明白了，我就照这个做。",
]

CONSTRAINTS = [
    "时间有点紧",
    "预算不想太高",
    "怕自己忘记",
    "不太确定先做哪一步",
    "最好不要太麻烦",
    "现在有点累",
    "今天事情比较多",
    "想尽量稳妥一点",
    "最好能省点时间",
    "不想影响别人",
]

FOLLOW_UPS = [
    "如果临时有变化怎么办？",
    "那第一步应该先做什么？",
    "有没有更简单一点的办法？",
    "这样会不会太麻烦？",
    "需要提前准备什么吗？",
    "如果时间不够呢？",
    "要不要再确认一遍？",
    "你觉得这样安排合理吗？",
]

CHANGE_REQUESTS = [
    "我刚才想了一下，可能要改一下计划。",
    "情况有点变化，时间比原来少。",
    "我突然发现预算要再压低一点。",
    "刚才那个方案可以，但我想轻松一点。",
    "如果换成明天处理，会不会更合适？",
    "我担心自己做不完，能不能拆成两步？",
]

EMOTION_LINES = [
    "我有点着急。",
    "我怕自己处理不好。",
    "我现在有点烦。",
    "我其实不太想麻烦别人。",
    "我担心到时候来不及。",
    "我感觉事情有点乱。",
]

TOPIC_GOALS: dict[str, list[str]] = {
    "买菜": ["确认购物清单", "控制买菜预算", "安排晚饭食材"],
    "做饭": ["安排今晚菜单", "把做饭步骤理清楚", "用现有食材做一顿饭"],
    "天气": ["决定出门穿什么", "判断要不要带伞", "安排周末外出"],
    "通勤": ["选择通勤方式", "避免迟到", "规划出门时间"],
    "快递": ["安排取快递", "确认快递内容", "处理大件包裹"],
    "约朋友": ["确定见面时间", "选择吃饭地点", "协调朋友安排"],
    "看病": ["判断要不要就医", "准备就诊事项", "安排休息和饮食"],
    "租房": ["筛选合适房源", "准备看房问题", "比较租房条件"],
    "学习": ["安排学习计划", "解决看不懂的问题", "提高复习效率"],
    "工作沟通": ["确认需求重点", "安排交付时间", "整理沟通事项"],
    "手机": ["延长手机电量", "处理手机卡顿", "安排充电"],
    "运动": ["安排轻松运动", "缓解久坐不适", "坚持锻炼"],
    "外卖": ["选择外卖", "控制配送时间", "点一顿清淡的饭"],
    "超市购物": ["列购物清单", "补充日用品", "比较价格"],
    "银行办事": ["准备办卡材料", "安排去银行时间", "确认账户信息"],
    "家务": ["分配家务", "整理房间", "安排清洁顺序"],
    "睡眠": ["改善睡眠", "安排早睡", "减少睡前干扰"],
    "旅行": ["规划短途旅行", "控制旅行预算", "准备行李"],
    "理发": ["预约理发", "决定发型", "选择理发店"],
    "宠物": ["照顾宠物饮食", "购买宠物用品", "安排陪宠物玩"],
    "邻里": ["处理噪音问题", "和邻居沟通", "联系物业"],
    "家庭": ["安排回家吃饭", "准备探望家人", "陪家人聊天"],
}

DETAILS = [
    "现在还没有完全想清楚。",
    "我想先听听你的建议。",
    "最好能给我一个具体步骤。",
    "我希望今天就能处理一部分。",
    "我不想把事情拖到太晚。",
    "我想先做最重要的部分。",
    "如果能简单一点就更好。",
    "我需要一个不容易出错的方案。",
]


def pick_topic_goal(topic: str, rng: random.Random) -> str:
    """为主题挑一个更具体的对话目标。"""

    return rng.choice(TOPIC_GOALS.get(topic, [f"处理{topic}相关的事情"]))


def build_synthetic_dialogue(topic: str, rng: random.Random) -> str:
    """生成一段原创中文日常对话。

    这个函数不只随机拼几句固定问答，还会随机选择一种“对话逻辑”：
    - basic：普通日常交流。
    - planning：先明确目标，再拆步骤。
    - troubleshooting：遇到问题，先定位原因，再给处理办法。
    - change：用户临时改主意，助手更新方案。
    - emotion：用户表达焦虑或犹豫，助手先安抚再建议。
    - compare：在两个选择之间做比较。

    这样生成的数据更容易覆盖真实对话里的追问、澄清、确认和调整。
    """

    pattern = rng.choice(["basic", "planning", "troubleshooting", "change", "emotion", "compare"])
    if pattern == "basic":
        return build_basic_dialogue(topic, rng)
    if pattern == "planning":
        return build_planning_dialogue(topic, rng)
    if pattern == "troubleshooting":
        return build_troubleshooting_dialogue(topic, rng)
    if pattern == "change":
        return build_change_dialogue(topic, rng)
    if pattern == "emotion":
        return build_emotion_dialogue(topic, rng)
    return build_compare_dialogue(topic, rng)


def build_basic_dialogue(topic: str, rng: random.Random) -> str:
    """普通日常交流：适合学习自然寒暄和直接问答。"""

    pairs = SCENARIOS[topic][:]
    rng.shuffle(pairs)
    selected = pairs[: rng.randint(3, len(pairs))]

    lines = [f"主题：{topic}", "对话类型：普通交流", f"用户：{rng.choice(OPENINGS)}", "助手：可以，我们慢慢说。"]
    for user_turn, assistant_turn in selected:
        lines.append(f"用户：{user_turn}")
        lines.append(f"助手：{assistant_turn}")
    lines.append(f"用户：{rng.choice(CLOSINGS)}")
    lines.append("助手：好的，有需要再随时说。")
    return "\n".join(lines)


def build_planning_dialogue(topic: str, rng: random.Random) -> str:
    """计划型对话：目标 -> 限制 -> 步骤 -> 确认。"""

    goal = pick_topic_goal(topic, rng)
    pair = rng.choice(SCENARIOS[topic])
    lines = [
        f"主题：{topic}",
        "对话类型：计划安排",
        f"用户：我想{goal}，但{rng.choice(CONSTRAINTS)}。",
        "助手：可以。我们先把目标、时间和限制说清楚，再拆成几步做。",
        f"用户：{rng.choice(DETAILS)}",
        "助手：第一步先确认最关键的那句话，不要同时处理太多信息。",
        f"用户：{pair[0]}",
        f"助手：{pair[1]}",
        f"用户：{rng.choice(FOLLOW_UPS)}",
        "助手：如果时间不够，就先做最影响结果的部分，剩下的放到第二步处理。",
        f"用户：{rng.choice(CLOSINGS)}",
        "助手：好，我帮你记住这个顺序：先确认重点，再处理必要事项，最后检查有没有遗漏。",
    ]
    return "\n".join(lines)


def build_troubleshooting_dialogue(topic: str, rng: random.Random) -> str:
    """问题排查型对话：描述问题 -> 追问原因 -> 给处理办法。"""

    pair = rng.choice(SCENARIOS[topic])
    lines = [
        f"主题：{topic}",
        "对话类型：问题排查",
        f"用户：我在{topic}这件事上遇到点问题，{rng.choice(EMOTION_LINES)}",
        "助手：先别急。你先说现在卡在哪一步，是信息不够、时间不够，还是不知道怎么选？",
        f"用户：主要是{rng.choice(CONSTRAINTS)}。",
        "助手：明白了。你再说一个具体情况，我帮你判断下一步。",
        f"用户：{pair[0]}",
        f"助手：{pair[1]}",
        f"用户：{rng.choice(FOLLOW_UPS)}",
        "助手：可以先做一个小决定，不用一次把所有问题都解决。做完第一步后，再根据结果调整。",
        "用户：这样听起来清楚一点。",
        "助手：对，日常问题很多时候不是一次解决，而是边确认边推进。",
    ]
    return "\n".join(lines)


def build_change_dialogue(topic: str, rng: random.Random) -> str:
    """变更型对话：原计划 -> 新限制 -> 更新安排。"""

    pairs = SCENARIOS[topic][:]
    rng.shuffle(pairs)
    first_pair = pairs[0]
    second_pair = pairs[1]
    lines = [
        f"主题：{topic}",
        "对话类型：临时变更",
        f"用户：我原本想按刚才的方式处理{topic}。",
        f"助手：可以。原计划里你先确认这件事：{first_pair[0]}",
        f"用户：{first_pair[0]}",
        f"助手：{first_pair[1]}",
        f"用户：{rng.choice(CHANGE_REQUESTS)}",
        "助手：那我们把计划调轻一点，先保留最必要的部分。",
        f"用户：具体怎么改？",
        f"助手：先处理这个问题：{second_pair[0]}",
        f"用户：{second_pair[0]}",
        f"助手：{second_pair[1]} 其他不急的可以放到明天或晚一点。",
        "用户：这样会不会影响结果？",
        "助手：影响不大。关键是先保证主要目标能完成，再处理细节。",
    ]
    return "\n".join(lines)


def build_emotion_dialogue(topic: str, rng: random.Random) -> str:
    """情绪支持型对话：先回应情绪，再给建议。"""

    goal = pick_topic_goal(topic, rng)
    pair = rng.choice(SCENARIOS[topic])
    lines = [
        f"主题：{topic}",
        "对话类型：情绪安抚",
        f"用户：我想{goal}，但是{rng.choice(EMOTION_LINES)}",
        "助手：能理解。先不用一下子想完，我们把事情拆小一点。",
        f"用户：我主要担心{rng.choice(CONSTRAINTS)}。",
        "助手：那我们先处理最可控的部分，不急着追求完美。",
        f"用户：比如呢？",
        f"助手：比如先把具体问题说出来：{pair[0]}",
        f"用户：{pair[0]}",
        f"助手：{pair[1]}",
        "用户：这样我感觉压力小一点。",
        "助手：对，先做一小步，事情就会从模糊变得清楚。",
    ]
    return "\n".join(lines)


def build_compare_dialogue(topic: str, rng: random.Random) -> str:
    """比较选择型对话：两个方案 -> 比较利弊 -> 做决定。"""

    pairs = SCENARIOS[topic][:]
    rng.shuffle(pairs)
    option_a = pairs[0]
    option_b = pairs[1]
    lines = [
        f"主题：{topic}",
        "对话类型：选择比较",
        f"用户：关于{topic}，我有两个选择，不知道哪个更合适。",
        "助手：可以，我们先比较时间、成本和麻烦程度。",
        f"用户：第一个选择是先问：{option_a[0]}",
        f"助手：这个选择比较稳，对应回应可以是：{option_a[1]}",
        f"用户：第二个选择是先问：{option_b[0]}",
        f"助手：这个选择更灵活，对应回应可以是：{option_b[1]}",
        f"用户：如果{rng.choice(CONSTRAINTS)}，选哪个？",
        "助手：那我建议选更稳的方案，先把主要问题解决，再考虑优化。",
        f"用户：{rng.choice(CLOSINGS)}",
        "助手：好，决定后就按步骤执行，别来回纠结太久。",
    ]
    return "\n".join(lines)


def iter_synthetic_zh(max_docs: int, seed: int) -> Iterable[str]:
    """批量生成原创中文日常对话。"""

    rng = random.Random(seed)
    topics = list(SCENARIOS)
    for index in range(max_docs):
        topic = topics[index % len(topics)]
        yield "来源：synthetic_zh_daily_dialogue\n" + build_synthetic_dialogue(topic, rng)


def write_source_note(path: Path, stats: dict[str, int]) -> None:
    """写语料来源说明。"""

    lines = [
        "# 日常对话语料来源说明",
        "",
        f"- 输出文本：`{path.name}`",
        "",
        "## 写入数量",
        "",
    ]
    for name, count in stats.items():
        lines.append(f"- `{name}`：`{count}` 段")

    lines.extend(["", "## 数据来源", ""])
    if stats.get("synthetic_zh_daily_dialogue", 0) > 0:
        lines.append("- `synthetic_zh_daily_dialogue`：本项目原创生成的简体中文日常生活对话，用于教学训练。")
    if stats.get("roskoN/dailydialog", 0) > 0:
        lines.append("- `roskoN/dailydialog`：DailyDialog 多轮日常对话数据，数据集页面说明许可为 CC BY-NC-SA 4.0，非商业使用。")
    if stats.get("HuggingFaceTB/everyday-conversations-llama3.1-2k", 0) > 0:
        lines.append("- `HuggingFaceTB/everyday-conversations-llama3.1-2k`：简单日常用户/助手对话数据，包含购物、交通、健康、学习等主题。")

    lines.extend(
        [
            "",
            "## 许可提醒",
            "",
            "原创简体中文语料可用于本项目学习训练。",
            "如果你额外启用 `dailydialog` 或 `everyday` 开源来源，请重新检查对应数据集页面的许可条款。",
            "",
        ]
    )
    path.with_suffix(".source.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and generate daily-life dialogue corpus.")
    parser.add_argument("--out", default="data/text/raw/daily_life_dialogue.txt")
    parser.add_argument(
        "--sources",
        default="synthetic",
        help="Comma-separated sources. Use synthetic for Simplified Chinese. Optional English sources: dailydialog,everyday.",
    )
    parser.add_argument("--max-synthetic", type=int, default=30000)
    parser.add_argument("--max-dailydialog", type=int, default=1000)
    parser.add_argument("--max-everyday", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_sources = {name.strip() for name in args.sources.split(",") if name.strip()}
    source_iters: list[tuple[str, Iterable[str]]] = []
    if "synthetic" in selected_sources:
        source_iters.append(("synthetic_zh_daily_dialogue", iter_synthetic_zh(args.max_synthetic, args.seed)))
    if "dailydialog" in selected_sources:
        source_iters.append(("roskoN/dailydialog", iter_dailydialog(args.max_dailydialog)))
    if "everyday" in selected_sources:
        source_iters.append(("HuggingFaceTB/everyday-conversations-llama3.1-2k", iter_everyday_conversations(args.max_everyday)))

    stats: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for source_name, documents in source_iters:
            written = 0
            try:
                for document in documents:
                    document = clean_text(document)
                    if not document:
                        continue
                    f.write(document)
                    f.write("\n\n")
                    written += 1
            except Exception as exc:
                print(f"跳过来源 {source_name}，原因：{type(exc).__name__}: {exc}", file=sys.stderr)
            stats[source_name] = written

    write_source_note(out_path, stats)

    print(f"Daily dialogue corpus saved to: {out_path}")
    print(f"Source note saved to: {out_path.with_suffix('.source.md')}")
    for source_name, count in stats.items():
        print(f"{source_name}: {count}")


if __name__ == "__main__":
    main()
