# 从零实现 Olmo 3 7B 与 32B

本文件夹中的[standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter 笔记本包含Olmo 3 7B与32B的从零实现，运行约需13 GB内存。

另一个[standalone-olmo3-plus-kv-cache.ipynb](standalone-olmo3-plus-kv-cache.ipynb)笔记本加入了KV缓存以提升运行时性能（但代码复杂度更高）。关于KV缓存的详细说明，可参阅文章[Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

下方给出了与Qwen3的对比图作为参考；如果对Qwen3 0.6B的独立笔记本感兴趣，可在[这里](../11_qwen3)找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3也有不同的“口味”，如下所示（架构相同，只是训练流程不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3与Qwen3的对比

本节聚焦架构而非训练细节，简要对比Qwen3。

7B模型：

1. 如上图所示，Olmo 3的架构与Qwen3相当类似。不过值得注意的是，这主要很可能延续自前代Olmo 2，而非直接来自Qwen3。

2) 与Olmo 2相似，Olmo 3仍采用post-norm（而非pre-norm），因为Olmo 2论文发现这样能稳定训练。

3) 有趣的是，7B仍使用类似Olmo 2的多头注意力；但为了更高效率并减小KV 缓存，它现在采用滑动窗口注意力（如 Gemma 3）。

接下来是32B模型：

4) 整体仍是同样的架构，只是规模放大。并且各比例（如前馈层输入到中间维度的扩展等）大致与Qwen3相符。

5) 猜测由于词表更小，最初架构可能比Qwen3略小；随后将中间层扩展系数从Qwen3的5倍提高到Olmo 3的5.4倍，以得到32B规模来做直接对比。

6) 另外，32B模型（终于！）使用了grouped query attention。




<br>

若想了解更多架构差异以及与其他模型的对比，参见文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。
