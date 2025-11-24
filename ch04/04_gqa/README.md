# 分组查询注意力 (GQA)

本额外材料展示了当使用分组查询注意力（GQA）替代常规多头注意力（MHA）时的内存节省。

&nbsp;
## 引言

近年来，GQA 已成为多头注意力（MHA）在计算与参数效率方面更优的替代方案。需要说明的是，它并非全新概念，可追溯到 2023 年的论文 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。甚至早期的 Llama 2 系列大型变体也使用了它。

简要来说，与每个头都拥有独立键和值的MHA不同，GQA将多个头分组，共享同一套键和值投影，以减少内存使用。

例如，如下图所示，如果有3组键值和6个注意力头，那么头1与2共享一套键和值，头3与4 以及头5与6各自共享另外两套。

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

这种键和值的共享减少了需要计算的键和值总量，从而降低内存占用并提升效率。

因此，总结起来，GQA的核心思想是让多个查询头共享键和值，从而减少键和值头的数量。这既（1）降低了模型的参数量，也（2）减少了推理阶段键值张量的内存带宽消耗，因为 KV 缓存中需要存取的键和值更少。

虽然GQA主要是面向MHA的计算效率权衡方案，但消融研究（如 [GQA原始论文](https://arxiv.org/abs/2305.13245)与[Llama 2论文](https://arxiv.org/abs/2307.09288)）表明，它在LLM的建模性能上与标准MHA相当。

不过，这建立在谨慎选择键值组数量的前提下。极端情况下，如果所有注意力头共享同一组键值（即多查询注意力），内存占用会进一步降低，但建模性能可能受损。（相反，如果键值组数量等于查询头数量，就回到了标准多头注意力。）

&nbsp;
## GQA的内存节省

内存节省主要体现在KV缓存。可以通过以下公式计算KV缓存的大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

可以使用本文件夹中的[memory_estimator_gqa.py](memory_estimator_gqa.py)脚本，对不同模型配置进行计算，查看使用GQA相比MHA可以节省多少内存：

```bash
uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步展示了在不同键值组数量下，GQA相对于MHA的内存节省如何随上下文长度变化：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

可以通过运行`uv run plot_memory_estimates_gqa.py`来复现这张图。

&nbsp;
## GQA代码示例

本文件夹中的[gpt_with_kv_mha.py](gpt_with_kv_mha.py)和[gpt_with_kv_gqa.py](gpt_with_kv_gqa.py)提供了比较GPT模型中MHA与GQA内存使用的示例。

请注意，GQA也被用于[Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3)与[Qwen3](../../ch05/11_qwen3)等额外材料。但为简洁起见，此文件夹中的脚本仅修改传统GPT架构，而经典GPT原本并未使用GQA。

此外，这里的模型未经过训练，因此会生成无意义的文本。不过可以在第5-7章中将其作为标准GPT模型的替代版本并训练。

该实现还使用了[另一个额外章节](../03_kv-cache)中介绍的KV缓存，因此能更加明显地体现内存节省。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

在这里看到的节省没有上图那么明显，主要有两点原因：

1. 为了在合理时间内完成生成，使用了更小的配置。
2. 更重要的是，查看的是整个模型而非单独的注意力机制；模型中的全连接层占据了大部分内存（这属于另一个分析主题）。
