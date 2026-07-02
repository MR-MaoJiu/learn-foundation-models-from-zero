# 学习文档导航

这个目录把项目拆成三条学习线。建议按顺序阅读，因为后面的内容会复用前面的概念。

## 1. LLM 文本模型

先读：

- `docs/llm/from_zero_principles.md`
- `docs/llm/implementation_guide.md`

你会学到：

- 文本如何变成 token。
- token 如何变成 embedding。
- Transformer 为什么能建模上下文。
- GPT 类模型为什么训练目标是预测下一个 token。

## 2. Vision 视觉模型

再读：

- `docs/vision/from_zero_principles.md`
- `docs/vision/implementation_guide.md`

你会学到：

- 图片如何变成张量。
- CNN 如何提取局部视觉特征。
- 分类模型如何用交叉熵训练。

## 3. Multimodal 多模态模型

最后读：

- `docs/multimodal/from_zero_principles.md`
- `docs/multimodal/implementation_guide.md`

你会学到：

- 图片和文本如何变成同一个向量空间里的表示。
- CLIP 风格的对比学习如何工作。
- 图文匹配为什么可以用相似度矩阵训练。

## 建议学习方式

每条线都按这个顺序：

```text
读原理文档 -> 读实现导读 -> 看 config -> 看 data.py -> 看 model.py -> 看训练脚本 -> 运行 count_params -> 小规模训练
```

不要急着改大模型。先跑通 debug 配置，再逐步改 batch size、层数、隐藏维度。
