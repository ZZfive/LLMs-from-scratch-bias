# 第4章：从零实现GPT模型以生成文本

&nbsp;
## 主章节代码

- [01_main-chapter-code](01_main-chapter-code)包含主章节代码。

&nbsp;
## 额外材料

- [02_performance-analysis](02_performance-analysis)包含可选代码，用于分析主章节中实现的GPT模型性能
- [03_kv-cache](03_kv-cache)实现KV缓存，以加速推理阶段的文本生成
- [07_moe](07_moe)讲解并实现Mixture-of-Experts(MoE)
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama)包含逐步讲解，演示如何将一个GPT架构实现转换为Llama 3.2，并加载Meta AI的预训练权重（完成第4章后查看替代架构会很有意思，你也可以等读完第5章再看）


&nbsp;
## 注意力机制的替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa)介绍Grouped-Query Attention(GQA)，这是大多数现代LLM（Llama 4、gpt-oss、Qwen3、Gemma 3等）替代常规Multi-Head Attention (MHA)的做法
- [05_mla](05_mla)介绍Multi-Head Latent Attention (MLA)，DeepSeek V3将其作为常规MHA的替代
- [06_swa](06_swa)介绍Sliding Window Attention (SWA)，Gemma 3等模型在使用
- [08_deltanet](08_deltanet)讲解Gated DeltaNet，这是一种广受欢迎的线性注意力变体（用于Qwen3-Next和Kimi Linear）


&nbsp;
## 更多内容

在下方视频中，提供了一次代码跟练，涵盖本章的部分内容，作为补充资料。

<br>
<br>

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
