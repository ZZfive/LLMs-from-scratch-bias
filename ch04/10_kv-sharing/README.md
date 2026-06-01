# 跨层 KV 共享

此补充材料说明了使用跨层 KV 共享与 KV 缓存一起时的内存节省情况。

&nbsp;
## 引言

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA），其中多个查询头共享相同的键和值头。跨层 KV 共享在变换器层之间应用了相关的想法。

Instead of computing a fresh key and value projection in every layer, later layers reuse K/V tensors from an earlier layer. They still compute their own queries, so each layer can form its own attention pattern. The main memory saving comes from storing fewer K/V tensors in the cache.
-> 而不是在每一层都计算新的键和值投影，后续层重用早期层的 K/V 张量。它们仍然计算自己的查询，因此每层可以形成自己的注意力模式。主要的内存节省来自于在缓存中存储更少的 K/V 张量。

This idea is also called cross-layer attention. It is described in Brandon *et al.*, [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981). Gemma 4 E2B and E4B use a related shared KV-cache scheme, which makes this a useful addition to the GQA, MLA, and SWA examples in this chapter.
-> 这个想法也被称为跨层注意力。它描述在 Brandon *et al.*, [使用跨层注意力减少 Transformer 键值缓存大小](https://arxiv.org/abs/2405.12981) 中。Gemma 4 E2B 和 E4B 使用相关的共享 KV 缓存方案，这使得它成为本章中 GQA、MLA 和 SWA 示例的有用补充。

&nbsp;

<img src="gemma4-kv-sharing.webp" alt="跨层 KV 共享" width="800px" />

&nbsp;

In [Gemma 4](../../ch05/17_gemma4), KV sharing is combined with GQA or MQA and sliding window attention. For the simplified GPT example in this folder, we only implement the cross-layer KV-sharing part, so the code stays focused on the main mechanism.
-> 在 [Gemma 4](../../ch05/17_gemma4) 中，KV 共享与 GQA 或 MQA 以及滑动窗口注意力相结合。对于本文件夹中简化的 GPT 示例，我们只实现跨层 KV 共享部分，因此代码专注于主要机制。

The simplified rule used here is:
-> 这里使用的简化规则是：
1. Early layers compute and cache their own K/V tensors.
-> 1. 早期层计算并缓存它们自己的 K/V 张量。
2. Later layers reuse the most recent K/V tensors from an earlier producing layer.
-> 2. 后续层重用早期产生层最新的 K/V 张量。
3. All layers still compute their own query projections.
-> 3. 所有层仍然计算它们自己的查询投影。

This reduces the number of K/V caches that grow with context length. The tradeoff is reduced model capacity because some layers no longer get their own K/V projections.
-> 这减少了随上下文长度增长的 K/V 缓存数量。权衡是模型容量降低，因为某些层不再获得它们自己的 K/V 投影。

&nbsp;
## KV 共享内存节省

The usual KV-cache memory is computed as follows:
-> 通常的 KV 缓存内存计算如下：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_layers x 2 (K,V) x bytes_per_elem

With cross-layer KV sharing, we replace `n_layers` with the number of K/V-producing layers:
-> 使用跨层 KV 共享，我们将 `n_layers` 替换为 K/V 产生层的数量：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_kv_producing_layers x 2 (K,V) x bytes_per_elem

You can use the [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) script in this folder to apply this to different model configs:
-> 您可以使用本文件夹中的 [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) 脚本将此应用于不同的模型配置：

```bash
# Gemma 4 E2B-like setup
uv run memory_estimator_kv_sharing.py \
  --context_length 131072 \
  --emb_dim 2048 \
  --n_heads 8 \
  --n_layers 35 \
  --n_kv_groups 8 \
  --n_kv_producing_layers 15 \
  --batch_size 1 \
  --dtype bf16

# Gemma 4 E4B-like setup
# uv run memory_estimator_kv_sharing.py \
#   --context_length 131072 \
#   --emb_dim 2560 \
#   --n_heads 8 \
#   --n_layers 42 \
#   --n_kv_groups 4 \
#   --n_kv_producing_layers 24 \
#   --batch_size 1 \
#   --dtype bf16

==== Config ====
context_length         : 131072
emb_dim                : 2048
n_heads                : 8
n_layers               : 35
n_kv_groups            : 8
n_kv_producing_layers  : 15
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 256
GQA n_kv_heads         : 1

==== KV-cache totals across all layers ====
MHA total KV cache        : 37.58 GB
GQA total KV cache        : 4.70 GB
MHA + KV sharing          : 16.11 GB
GQA + KV sharing          : 2.01 GB
Ratio (MHA / GQA+sharing) : 18.67x
Savings vs MHA            : 94.64%
```

This is a Gemma 4 E2B-like setup. The 35 layers include 15 K/V-producing layers, and the remaining layers reuse earlier K/V tensors. For the E4B-like setup, the corresponding numbers are 42 total layers and 24 K/V-producing layers.
-> 这是一个类似 Gemma 4 E2B 的设置。35 层包括 15 个 K/V 产生层，其余层重用早期的 K/V 张量。对于类似 E4B 的设置，相应的数字是 42 个总层和 24 个 K/V 产生层。

The savings are shown below for the E2B-like and E4B-like setups. For simplicity, these plots do not include additional savings from sliding window attention.
-> 下面显示了类似 E2B 和类似 E4B 设置的节省情况。为简单起见，这些图不包括滑动窗口注意力带来的额外节省。

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e2b.webp" alt="Gemma 4 E2B 类似设置的 KV 共享内存节省" width="800px" />

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e4b.webp" alt="Gemma 4 E4B 类似设置的 KV 共享内存节省" width="800px" />

&nbsp;

You can reproduce similar plots via:
-> 您可以通过以下方式重现类似的图：

```bash
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e2b
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e4b
```

&nbsp;
## KV 共享代码示例

The [gpt_with_kv_mha.py](gpt_with_kv_mha.py) and [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) scripts in this folder provide hands-on examples for comparing regular MHA with a cross-layer KV-sharing variant.
-> 本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 脚本提供了手动比较常规 MHA 与跨层 KV 共享变体的示例。

The easiest way to see the implementation details is to inspect a file diff between [gpt_with_kv_mha.py](gpt_with_kv_mha.py) and [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py). The comments are intentionally kept similar so that the diff highlights the KV-sharing changes.
-> 查看实现细节的最简单方法是检查 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 之间的文件差异。注释故意保持相似，以便差异突出显示 KV 共享的更改。

Note that the model is not trained and thus generates nonsensical text. However, you can use it as a drop-in replacement for the standard GPT model in chapters 5-7 and train it.
-> 注意该模型未经训练，因此生成的文本无意义。但是，您可以将其作为第 5-7 章中标准 GPT 模型的替代品并对其进行训练。

Also, this implementation uses the KV cache explained in [another bonus section](../03_kv-cache), so the memory savings are more pronounced.
-> 此外，此实现使用了 [另一个补充部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更为明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768
```

```bash
uv run gpt_with_kv_sharing.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--n_kv_producing_layers 6
```

In this small GPT setup, the whole model still contains the same feed-forward layers and output head. The main memory difference is in how many attention layers store K/V tensors in the cache.
-> 在这个小型 GPT 设置中，整个模型仍然包含相同的前馈层和输出头。主要内存差异在于多少注意力层在缓存中存储 K/V 张量。
