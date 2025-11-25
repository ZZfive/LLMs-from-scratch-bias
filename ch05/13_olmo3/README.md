# 从零实现 Olmo 3 7B 与 32B

本文件夹中的 [standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter 笔记本包含 Olmo 3 7B 与 32B 的从零实现，运行约需 13 GB 内存。

另一个 [standalone-olmo3-plus-kv-cache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) 笔记本加入了 KV 缓存以提升运行时性能（但代码复杂度更高）。关于 KV 缓存的详细说明，可参阅我的文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

下方给出了与 Qwen3 的对比图作为参考；如果你对 Qwen3 0.6B 的独立笔记本感兴趣，可在 [这里](../11_qwen3) 找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3 也有不同的“口味”，如下所示（架构相同，只是训练流程不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3 与 Qwen3 的对比

本节聚焦架构而非训练细节，简要对比 Qwen3。

7B 模型：

1. 如上图所示，Olmo 3 的架构与 Qwen3 相当类似。不过值得注意的是，这主要很可能延续自前代 Olmo 2，而非直接来自 Qwen3。

2) 与 Olmo 2 相似，Olmo 3 仍采用 post-norm（而非 pre-norm），因为 Olmo 2 论文发现这样能稳定训练。

3) 有趣的是，7B 仍使用类似 Olmo 2 的多头注意力；但为了更高效率并减小 KV 缓存，它现在采用滑动窗口注意力（如 Gemma 3）。

接下来是 32B 模型：

4) 整体仍是同样的架构，只是规模放大。并且各比例（如前馈层输入到中间维度的扩展等）大致与 Qwen3 相符。

5) 我猜由于词表更小，最初架构可能比 Qwen3 略小；随后他们将中间层扩展系数从 Qwen3 的 5 倍提高到 Olmo 3 的 5.4 倍，以得到 32B 规模来做直接对比。

6) 另外，32B 模型（终于！）使用了 grouped query attention。




<br>

若想了解更多架构差异以及与其他模型的对比，参见我的文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。
