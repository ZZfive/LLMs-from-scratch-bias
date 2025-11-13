# Gemma 3 270M 从零开始

此文件夹中的[standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter笔记本包含Gemma 3 270M的从零实现。它需要大约2 GB的RAM来运行。

替代的[standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb)笔记本增加了KV缓存以获得更好的运行时性能（但增加了更多代码复杂性）。要了解有关KV缓存的更多信息，请参阅我的文章[从零理解和编码LLM中的KV缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

| 模型             | 模式              | 硬件          | Tokens/sec | GPU内存（VRAM） |
| ---------------- | ----------------- | ------------- | ---------- | --------------- |
| Gemma3Model 270M | 常规版            | Mac Mini M4 CPU | 8          | -               |
| Gemma3Model 270M | 常规版编译        | Mac Mini M4 CPU | 9          | -               |
| Gemma3Model 270M | KV缓存            | Mac Mini M4 CPU | 130        | -               |
| Gemma3Model 270M | KV缓存编译        | Mac Mini M4 CPU | 224        | -               |
|                  |                   |               |            |                 |
| Gemma3Model 270M | 常规版            | Mac Mini M4 GPU | 16         | -               |
| Gemma3Model 270M | 常规版编译        | Mac Mini M4 GPU | 错误        | -               |
| Gemma3Model 270M | KV缓存            | Mac Mini M4 GPU | 23         | -               |
| Gemma3Model 270M | KV缓存编译        | Mac Mini M4 GPU | 错误        | -               |
|                  |                   |               |            |                 |
| Gemma3Model 270M | 常规版            | Nvidia A100 GPU | 28         | 1.84 GB           |
| Gemma3Model 270M | 常规版编译        | Nvidia A100 GPU | 128        | 2.12 GB           |
| Gemma3Model 270M | KV缓存            | Nvidia A100 GPU | 26         | 1.77 GB           |
| Gemma3Model 270M | KV缓存编译        | Nvidia A100 GPU | 99         | 2.12 GB           |


以下是作为参考模型与Qwen3 0.6B的并排比较；如果您对Qwen3 0.6B独立笔记本感兴趣，可以在这里找到[这里](../11_qwen3)。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

要了解有关架构差异的更多信息并阅读与其他架构的比较，请参阅我的文章[大LLM架构比较：从DeepSeek-V3到Kimi K2：现代LLM架构设计一览](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。





