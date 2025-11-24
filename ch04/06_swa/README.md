# 滑动窗口注意力 (SWA)

本额外材料展示了当使用滑动窗口注意力（SWA）取代常规多头注意力（MHA）时的内存节省。



&nbsp;
## 引言

什么是滑动窗口注意力（SWA）？如果把常规自注意力看作 *全局* 注意力机制（因为序列中每个元素都能访问其他所有元素），那么SWA可以视为*局部*注意力，因为它限制了当前查询位置周围的上下文大小，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如图所示，每个token只关注其位置周围的固定大小窗口，而不是关注所有先前token。这种局部注意力大幅降低了KV缓存的大小。

下面的介绍中，将结合[Gemma 3](https://arxiv.org/abs/2503.19786)（在 [../../ch05/12_gemma3](../../ch05/12_gemma3)中从零实现）来讨论SWA。

滑动窗口注意力最初在[2020年的LongFormer论文](https://arxiv.org/abs/2004.05150)中提出，不过关注Google的Gemma系列，是因为它们是性能优异的开源权重模型，证明了滑动窗口注意力在最新的高性能模型中是可行的做法。

[Gemma 2](https://arxiv.org/abs/2408.00118)使用了局部（滑动窗口）与全局注意力层1:1结合的混合策略。每个token可以关注4k个token的上下文。采用1:1混合是为了在效率与全局上下文建模之间取得平衡，因为仅使用局部注意力会限制过多。

[Gemma 3](https://arxiv.org/abs/2503.19786)在设计上进一步追求效率。它采用了5:1的滑动窗口与全局注意力比例：每五层局部注意力后跟一层全局注意力。此外，滑动窗口大小也从Gemma 2中的4096 tokens减少到1024 tokens。

有趣的是，Gemma 3技术报告中的消融实验表明，这些变化对整体模型质量影响很小。换句话说，滑动窗口注意力带来的巨大内存和计算节省，只会造成极小的性能损失。



&nbsp;
## 滑动窗口注意力 (SWA) 的内存节省

内存节省主要体现在KV缓存。可以通过以下公式计算KV存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

当使用SWA时，将上述公式中的序列长度seqlen替换为窗口大小W。因此，滑动窗口注意力可按“W / seqlen”的比例减少KV缓存大小。（为简单起见，这里假设每一层都使用滑动窗口注意力。）


可以使用本文件夹中的[memory_estimator_swa.py](memory_estimator_swa.py)来测试不同模型配置，了解使用SWA相比MHA能节省多少内存：

```bash
uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
MHA + GQA (Ratio: 5:1) : 0.78 GB
```

需要注意，Gemma 3将SWA与GQA结合使用。

下图展示了在不同上下文长度下，SWA相比MHA的节省情况：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

可以运行以下命令复现这些图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA代码示例

本文件夹中的[gpt_with_kv_mha.py](gpt_with_kv_mha.py)与[gpt_with_kv_swa.py](gpt_with_kv_swa.py)提供了GPT模型实现中比较MHA与SWA内存使用的示例。

请注意，SWA也可以与MLA、GQA（前面提过）结合使用，但这里为简化没有演示。

此外，这里的模型并未经过训练，因此会生成无意义的文本。不过可以在第5-7章中把它作为标准GPT模型的替代版本并进行训练。

同时，该实现使用了[另一个额外章节](../03_kv-cache)中介绍的KV缓存，因此内存节省更为明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # 类似 Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

这里看到的节省没有上图那么显著，原因主要有两点：

1. 使用了较小的配置，以便模型能在合理时间内完成生成。
2. 更重要的是，关注的是整个模型，而非单独的注意力机制；模型中的全连接层占据了大部分内存（这需要单独分析）。
