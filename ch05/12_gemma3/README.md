# 从零构建 Gemma 3 270M

此文件夹中的 [standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter notebook 包含 Gemma 3 270M 的从零实现，运行大约需要 2 GB 内存。

另一个 [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) notebook 在此基础上加入了 KV 缓存，以获得更好的运行时性能（但代码也会更复杂）。如果你想进一步了解 KV 缓存，可以参阅我的文章[从零理解和实现 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

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


下图给出了与参考模型 Qwen3 0.6B 的并排比较；如果你对 Qwen3 0.6B 的独立 notebook 感兴趣，可以在[这里](../11_qwen3)找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

如果你想进一步了解架构差异，以及与其他架构的对比，可以参阅我的文章[大语言模型架构全景对比：从 DeepSeek-V3 到 Kimi K2](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。





