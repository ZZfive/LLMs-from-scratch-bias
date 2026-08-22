# Gemma 4

本目录包含一个独立的纯文本 Gemma 4 笔记本，该笔记本基于 Gemma 3 参考笔记本构建，并适配用于密集型的 `google/gemma-4-E2B` 和 `google/gemma-4-E4B` 检查点。

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) 使用纯 PyTorch 实现了共享的 Gemma 4 密集架构，并通过 `CHOOSE_MODEL` 在 E2B 和 E4B 参考配置之间进行切换。
