# 跨层 KV 共享

本补充材料说明了结合 KV 缓存使用跨层 KV 共享时的内存节省情况。

&nbsp;
## 引言

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA），其中几个查询头共享相同的键和值头。跨层 KV 共享将相关理念应用于 Transformer 层之间。

与其在每一层都计算新的键和值投影，不如后来的层复用早期层的 K/V 张量。它们仍然计算自己的查询，因此每层可以形成自己的注意力模式。主要的内存节省来自于在缓存中存储更少的 K/V 张量。

这个想法也被称为跨层注意力。Brandon 等人在 [使用跨层注意力减少 Transformer 键值缓存大小](https://arxiv.org/abs/2405.12981) 中对这一方法进行了描述。Gemma 4 E2B 和 E4B 使用了相关的共享 KV 缓存方案，这使得本章的 GQA、MLA 和 SWA 示例有了一个有用的补充。

&nbsp;

<img src="gemma4-kv-sharing.webp" alt="跨层 KV 共享" width="800px" />

&nbsp;

在 [Gemma 4](../../ch05/17_gemma4) 中，KV 共享与 GQA 或 MQA 以及滑动窗口注意力相结合。对于本文件夹中的简化 GPT 示例，我们仅实现跨层 KV 共享部分，以便代码专注于主要机制。

此处使用的简化规则是：

1. 早期层计算并缓存自己的 K/V 张量。
2. 后来的层复用早期生成层最近的 K/V 张量。
3. 所有层仍然计算自己的查询投影。

这减少了随上下文长度增长的 K/V 缓存数量。代价是模型容量降低，因为某些层不再拥有自己的 K/V 投影。

&nbsp;
## KV 共享的内存节省

通常的 KV 缓存内存计算如下：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_layers x 2 (K,V) x bytes_per_elem

使用跨层 KV 共享时，我们将 `n_layers` 替换为 K/V 生成层的数量：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_kv_producing_layers x 2 (K,V) x bytes_per_elem

您可以使用本文件夹中的 [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) 脚本将其应用于不同的模型配置：

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

这是一个类似 Gemma 4 E2B 的设置。35 层中包含 15 个 K/V 生成层，其余层复用早期的 K/V 张量。对于类似 E4B 的设置，相应的数字是总共 42 层和 24 个 K/V 生成层。

下文展示了类似 E2B 和 E4B 设置的节省情况。为简化起见，这些图表未包含滑动窗口注意力带来的额外节省。

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e2b.webp" alt="类似 Gemma 4 E2B 设置的 KV 共享内存节省" width="800px" />

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e4b.webp" alt="类似 Gemma 4 E4B 设置的 KV 共享内存节省" width="800px" />

&nbsp;

您可以通过以下方式重现类似的图表：

```bash
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e2b
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e4b
```

&nbsp;
## KV 共享代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 脚本提供了比较常规 MHA 与跨层 KV 共享变体的实际操作示例。

查看实现细节的最简单方法是检查 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 之间的文件差异。注释有意保持相似，以便差异突出显示 KV 共享的变更。

请注意，该模型未经过训练，因此会生成无意义的文本。不过，您可以将其用作第 5-7 章中标准 GPT 模型的直接替代品并进行训练。

此外，此实现使用了 [另一个补充部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更为显著。

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

在这个小型 GPT 设置中，整个模型仍然包含相同的前馈层和输出头。主要的内存区别在于有多少注意力层在缓存中存储 K/V 张量。
