# 多头潜在注意力 (MLA)

本额外材料展示了采用多头潜在注意力（MLA）替代常规多头注意力（MHA）时的内存节省。

&nbsp;
## 引言

在 [../04_gqa](../04_gqa) 中，讨论了将分组查询注意力（GQA）作为提升MHA计算效率的方案。而消融实验（例如[GQA原始论文](https://arxiv.org/abs/2305.13245)和[Llama 2论文](https://arxiv.org/abs/2307.09288)）表明，它在大型语言模型的建模性能上与标准MHA相当。

而被[DeepSeek V2、V3与R1](https://arxiv.org/abs/2412.19437)采用的多头潜在注意力（MLA）提供了另一种配合KV缓存的内存节省策略。它并不像GQA那样让多个头共享键值，而是在将键、值存入KV缓存前先压缩到低维空间。 

在推理时，如下图所示，这些压缩后的张量会先投射回原尺寸再参与计算。虽然多了一次矩阵乘法，但可以减少内存使用。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（顺带一提，查询张量在训练阶段也会被压缩，但推理阶段不会。）

顺便提一下，MLA并非DeepSeek V3才出现的概念，其前代[DeepSeek V2](https://arxiv.org/abs/2405.04434)就已经使用甚至引入了它。V2论文中的一些消融实验也许能解释DeepSeek团队为何选择MLA而非GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA的表现不如MHA，而MLA的建模性能优于MHA，这可能是DeepSeek团队选择MLA而非GQA的原因。（如果能看到MLA与GQA的“每个token的KV缓存”节省对比就更有意思了！）

在深入下一部分结构组件之前，总结一下：MLA是一个巧妙的技巧，既能降低KV缓存的内存占用，还能在建模性能上略优于MHA。

&nbsp;
## MLA的内存节省

内存节省主要体现在KV存储上。可以通过以下公式计算KV存储大小：

bytes 鈮?batch_size 脳 seqlen 脳 n_layers 脳 latent_dim 脳 bytes_per_elem

相比之下，MHA 的 KV 缓存内存计算方式为：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

这意味着，在MLA中，将 “embed_dim × 2 (K,V)” 降低为 “latent_dim”，因为只需在KV缓存中存储压缩后的潜在表示，而非完整的键、值向量。



可以使用本文件夹中的[memory_estimator_mla.py](memory_estimator_mla.py)对不同模型配置进行计算，了解使用MLA相比MHA能节省多少内存：

```bash
鉃?uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

需要注意，上述配置中将`--emb_dim 2048`压缩到`latent_dim 1024`，以获得与GQA相近的内存节省。在实践中，压缩程度是需要仔细调优的超参数，因为将`latent_dim`设得过小会像GQA里`n_kv_groups`太多一样，对建模性能产生负面影响。

下图展示了不同`latent_dim`取值下，MLA相较MHA的内存节省随上下文长度的变化情况：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

可以运行`uv run plot_memory_estimates_mla.py`来复现此图。



&nbsp;
## MLA代码示例

本文件夹中的[gpt_with_kv_mha.py](gpt_with_kv_mha.py)与[gpt_with_kv_mla.py](gpt_with_kv_mla.py)提供了GPT模型实现中对比MHA与MLA内存使用的实战示例。 

其中的MLA实现借鉴了 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla)。

请注意，MLA也可以与[GQA](../04_gqa)结合使用，但为保持简洁，这里没有这么做。（目前也还未见到主流LLM这样组合。）

此外，这里的模型并未训练，因此会生成无意义的文本。不过你可以将其作为第5-7章标准GPT模型的替代版本并展开训练。

最后，该实现使用了[另一额外章节](../03_kv-cache)中介绍的KV缓存，因此内存节省更加明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× 压缩

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

在这里看到的内存节省没有上图那样巨大，原因有两点：

1. 使用了较小的配置，以便模型能在合理时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而非单独的注意力机制；模型中的全连接层占据了大部分内存（这有待另行分析）。

