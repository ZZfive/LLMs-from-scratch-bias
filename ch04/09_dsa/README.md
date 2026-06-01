# DeepSeek 稀疏注意力 (DSA)

本补充材料实现了在 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中引入的 DeepSeek 稀疏注意力 (DSA) 机制，并首次发布在实验性的 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 版本中。

以下概述遵循 [From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates](https://magazine.sebastianraschka.com/p/technical-deepseek) 中的 DSA 讨论。

&nbsp;
## 引言

标准因果自注意力为每个查询关注所有之前的标记，产生 O(L²) 的计算量和 O(L) 的 KV 缓存增长，随序列长度 L 变化。

[滑动窗口注意力 (SWA)](../06_swa) 已经表明，将注意力限制在固定的局部窗口上能显著降低此成本。在 SWA 中，每个查询标记仅关注附近之前标记的局部跨度。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="滑动窗口注意力" width="800px" />

*图 1. 滑动窗口注意力将每个查询标记限制在固定的局部上下文窗口内。*

&nbsp;

DSA 使用相同的广义思想，即只关注之前标记的一个子集。然而，它用学习到的选择机制替换了固定窗口。对于每个查询标记，模型对候选过去标记进行评分，并只保留最相关的标记。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek 稀疏注意力选择标记模式" width="800px" />

*图 2. DeepSeek 稀疏注意力为每个查询标记选择学习到的过去标记子集。*

&nbsp;

### 架构概述

DSA 在标准注意力之上增加了两个组件。

**1. 闪电索引器**

对于每个查询标记 $t$ 和每个候选过去标记 $s$，索引器计算一个标量相关性得分。此实现使参考代码中的缩放因子显式化：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量级索引头的数量，
- $q_{t,j}$ 是标记 $t$ 和头 $j$ 的索引器查询向量，
- $k_s$ 是过去标记 $s$ 的共享索引器键向量，
- $w_{t,j}$ 是一个学习到的每头门控，缩放比例为 $1 / \sqrt{H_I}$。

ReLU 将负点积贡献置零，门控求和将索引头聚合为每个过去标记的单个相关性得分。

在完整的 DeepSeek 模型中，索引器使用多头潜在注意力 (MLA) 的压缩标记表示。此文件夹保持 GPT 实现更简单，从常规隐藏状态计算索引器查询和键。

**2. 标记选择器**

计算完所有索引器得分后，仅保留得分最高的前 K 个位置。所有其他位置在标准 softmax 之前被掩码为 −∞，因此模型实际上只关注 $k \ll L$ 个标记。

索引器中的 ReLU 不是最终稀疏性的来源。由于得分是在多个索引头上求和，大多数最终得分仍可能为非零。标记选择器通过仅保留前 K 个位置来创建稀疏模式。

在融合的生产实现中，这可以将注意力计算从 O(L²) 降低到 O(L·k)。此实现保持标准密集注意力得分矩阵，并在 softmax 之前应用 DSA 选定的前 K 个掩码。这使得选择逻辑易于检查，但它不提供融合内核的计算节省。

下图总结了流程。闪电索引器对候选标记进行评分，选择器保留前 K 个位置， resulting mask 限制通常的注意力 softmax。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek 稀疏注意力流程图" width="700px" />

*图 3. DSA 首先对候选标记进行评分，然后为最终注意力标记保留前 K 个标记。*

&nbsp;
## 实现

`gpt_with_kv_dsa.py` 提供：

| 类 | 描述 |
|---|---|
| `LightningIndexer` | 用于过去标记相关性的轻量级多头评分器。 |
| `MultiHeadAttentionWithDSA` | 带有 DSA 稀疏掩码 + 可选 KV 缓存的标准 MHA。 |
| `GPTModel` | 在 `MultiHeadAttentionWithDSA` 中交换的 GPT 风格模型。 |

此实现遵循本仓库中其他补充材料的风格，可以作为独立脚本运行。它旨在使 DSA 机制在小 GPT 风格模型中可检查。它未实现 DeepSeek 的完整 MLA 堆栈、融合稀疏内核或部署特定优化。

&nbsp;
## 用法

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

关键参数：

| 参数 | 默认值 | 描述 |
|---|---|---|
| `--index_n_heads` | 4 | 轻量级索引头的数量 (H_I)。 |
| `--index_head_dim` | 64 | 每个索引头的维度。 |
| `--topk` | 64 | 每个查询关注的标记数量 (k)。对于短序列，上限为序列长度。 |

&nbsp;
## 与 DeepSeek V3.2 的关系

全规模的 DeepSeek-V3.2 模型使用多头潜在注意力 (MLA，见 [../05_mla](../05_mla)) 与 DSA 结合，且索引器查询源自共享的压缩潜在表示而非原始输入。DeepSeek-V3.2 使用与 DeepSeek-V3.2-Exp 相同的架构，其中 DSA 首次被引入和测试。

关键选择思想在此处被复现。一个廉价的点积评分器在注意力 softmax 之前将每个查询限制在最相关的标记上。

下面报告的推理成本比较提供了为什么 DSA 在长上下文部署中很重要的有用背景。节省取决于生产内核和服务基础设施，因此此图不应被视为本文件夹中教学实现的基准。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="DeepSeek 稀疏注意力推理成本比较" width="800px" />

*图 4. DeepSeek 在长上下文服务中从 DSA 报告的推理成本节省，来自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

&nbsp;
## 参考文献

- DeepSeek V3.2 技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp 模型卡及参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的 "From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates"：https://magazine.sebastianraschka.com/p/technical-deepseek
