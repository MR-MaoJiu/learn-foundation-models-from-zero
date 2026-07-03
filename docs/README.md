# 文档导航

这个仓库只保留 LLM 聊天模型主线。

推荐阅读顺序：

1. [LLM 原理](llm/from_zero_principles.md)
2. [代码导读](llm/implementation_guide.md)
3. [完整训练到部署流程](llm/full_workflow.md)

你会看到一条完整链路：

```text
语料准备 -> tokenizer -> 预训练 base model -> SFT chat model -> 评估 -> 导出 -> HTTP 部署
```

核心原则：

- 预训练学习通用文本分布。
- SFT 学习助手式问答行为。
- tokenizer、base checkpoint、SFT checkpoint 和部署包必须来自同一条训练链路。
