# 混合专家 (MoE)

本额外材料展示了在使用混合专家（MoE）层取代常规前馈（FFN）层时，每个 token 的内存节省。



&nbsp;
## 引言

MoE的核心思想是在每个transformer模块的前馈层中引入多个专家层，每个专家层本质上仍是一个前馈模块。这意味着会像下图所示那样，用多个前馈块替换单个前馈块。



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

transformer模块中的前馈块（上图中的深灰色部分）通常包含模型参数的大部分。（需要注意，transformer模块以及前馈块在LLM中会重复很多次，比如DeepSeek-V3重复了61次。）

因此，用多个前馈块替换单个前馈块会显著增加模型的总参数量。不过关键在于不会对每个token激活所有专家，而是由一个路由器来选择一个小的专家子集。

由于同时只有少数专家被激活，MoE模块通常被称为*稀疏*模块，以区别于始终使用全部参数的*稠密*模块。然而，通过MoE引入的大量参数增加了LLM的容量，使其在训练期间能够学习到更多知识。稀疏性则让推理仍然高效，因为不会在同一时间使用全部参数。

以DeepSeek-V3为例：每个MoE模块都包含256个专家，总参数达到6710亿。但在推理中只有9个专家同时激活（1个共享专家加上路由器选出的8个），这意味着每步推理实际只用了约370亿参数，而不是全部6710亿。

DeepSeek-V3的MoE设计中有一个值得注意的特点：共享专家。这是一个对所有token都始终激活的专家。在[2022年的DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)和[2024年的DeepSeek MoE](https://arxiv.org/abs/2401.06066)论文中就已经提出了这一点。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（来自论文 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 的注释图）

&nbsp;

共享专家的好处最早在[DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596)中被提出，该论文发现相比没有共享专家的情况，引入共享专家可以提升整体建模性能。原因可能是常见或重复的模式无需多个独立专家分别学习，从而留下更多空间来学习特化模式。

&nbsp;
## 混合专家(MoE)的内存节省

MoE的内存节省主要来自激活存储和计算的减少。在常规（稠密）前馈层中，每个token都会激活完整的中间维度。相比之下，MoE仅将每个token路由到部分专家（例如`num_experts`中的`top_k`个）。

因此，在MoE中，每个token只激活`top_k`个专家，其有效内存和计算量相当于将同样总容量的稠密FFN缩减了`top_k / num_experts`倍。


可以使用本文件夹中的[memory_estimator_moe.py](memory_estimator_moe.py)脚本，在不同模型配置下对比MoE与FFN的内存使用情况（注意此脚本针对单个transformer模块，如需总节省，需要乘以模型中的模块数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

从上面的结果可以看到，如果FFN的输入/输出维度(`emb_dim`)为7168，中间维度(`hidden_dim`)为14336，那么该层约有3.08亿个参数，并且在前向过程中全部参与计算。

如果使用参数量相似（同样约 3.08 亿）的MoE层，并设置8个专家、每次激活2个专家，那么每次前向只需激活约7700万个参数。

另外，在保持`top_k`不变的情况下，专家数量越多，每次激活的参数就越少，“节省”也越明显：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

可以运行以下命令复现该图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE代码示例

本文件夹中的[gpt_with_kv_ffn.py](gpt_with_kv_ffn.py)与[gpt_with_kv_moe.py](gpt_with_kv_moe.py)提供了GPT模型实现中常规FFN与MoE内存使用的比较示例。需要注意，这两个脚本都采用了[SwiGLU](https://arxiv.org/abs/2002.05202)前馈模块（如本文开头的图所示；GPT-2传统上使用GELU）。

**注意：模型并未训练，因此会生成无意义的文本。你可以在 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 中找到一个已训练的MoE示例。**



首先运行带有常规 FFN 的模型：

```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了公平对比MoE，需要减小每个专家的隐藏维度。例如若使用32个专家，需要设置 `--hidden_dim 32768/32`：

```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

可以看到，稠密前馈层处理一个token约需0.76 ms，激活使用约0.19 MB（峰值约0.75 MB）。

稀疏MoE层仅使用约0.04 MB（峰值0.11 MB），但代价是计算时间约翻倍。（路由过程增加了开销，而且此处实现也未必最优。）

总的生成阶段两者都需要约11.5 GB的GPU内存，因为两者加载的权重数和KV缓存大小相同，这些占用了主要内存。

可以看出，在处理前馈时MoE将内存降低了约4-5倍，但前馈计算时间增加了一倍左右。

需要注意，如果一次处理更多token（例如更大的batch，这里为了简化代码未使用batch），节省会更加明显。
