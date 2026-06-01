# Gemma 4

此目录包含一个独立的、纯文本的 Gemma 4 笔记本，该笔记本基于 Gemma 3 参考笔记本构建，并针对稠密 `google/gemma-4-E2B` 和 `google/gemma-4-E4B` 检查点进行了调整。

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) 使用纯 PyTorch 实现了共享的 Gemma 4 稠密架构，并通过 `CHOOSE_MODEL` 在 E2B 和 E4B 参考配置之间切换。
