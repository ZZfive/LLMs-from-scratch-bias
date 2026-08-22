# DeepSeek 稀疏注意力 (DSA)

本补充材料实现了 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中引入的 DeepSeek 稀疏注意力 (DSA) 机制，并首次在实验性 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 版本中发布。

下面的概述遵循了 [从 DeepSeek V3 到 V3.2：架构、稀疏注意力与 RL 更新](https://magazine.sebastianraschka.com/p/technical-deepseek) 中的 DSA 讨论。

&nbsp;
## 引言

标准的因果自注意力机制中，每个查询 token 会关注所有历史 token，导致计算复杂度为 O(L²)，且 KV 缓存 (KV-cache) 随序列长度 L 呈 O(L) 增长。

[滑动窗口注意力 (SWA)](../06_swa) 已经表明，将注意力限制在固定的局部窗口可以显著降低此成本。在 SWA 中，每个查询 token 仅关注附近历史 token 的局部跨度。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="滑动窗口注意力" width="800px" />

*图 1. 滑动窗口注意力将每个查询 token 限制在固定的局部上下文窗口内。*

&nbsp;

DSA 使用了类似的仅关注历史 token 子集的思想。然而，它用一种学习到的选择机制取代了固定窗口。对于每个查询 token，模型会对候选的历史 token 进行评分，并仅保留最相关的 token。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek 稀疏注意力选中的 token 模式" width="800px" />

*图 2. DeepSeek 稀疏注意力为每个查询 token 选择一个学习到的历史 token 子集。*

&nbsp;

### 架构概览

DSA 在标准注意力之上增加了两个组件。

**1. Lightning Indexer**

对于每个查询 token $t$ 和每个候选历史 token $s$，索引器计算一个标量相关性得分。此实现使参考代码中的缩放因子显式化：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量级索引头的数量，
- $q_{t,j}$ 是 token $t$ 和头 $j$ 的索引器查询向量，
- $k_s$ 是历史 token $s$ 的共享索引器 key 向量，
- $w_{t,j}$ 是按头学习的门控，按 $1 / \sqrt{H_I}$ 缩放。

ReLU 将负的点积贡献置零，并且门控求和在索引头之间聚合，为每个历史 token 生成单个相关性得分。

在完整的 DeepSeek 模型中，索引器与来自多头潜在注意力 (MLA) 的压缩 token 表示一起工作。此文件夹简化了 GPT 实现，并从常规隐藏状态计算索引器查询和 keys。

**2. Token Selector**

计算完所有索引器得分后，仅保留得分最高的前 K 个位置。所有其他位置在标准 softmax *之前* 被掩码为 −∞，因此模型实际上只关注 $k \ll L$ 个 token。

索引器中的 ReLU 并不是最终稀疏性的来源。由于得分是在多个索引头上求和的，大多数最终得分仍然可以是非零的。Token 选择器通过仅保留前 K 个位置来创建稀疏模式。

在生产环境的融合实现中，这可以将注意力计算从 O(L²) 降低到 O(L·k)。此处的实现保留了标准的稠密注意力分数矩阵，并在 softmax 之前应用 DSA 选择的前 K 个 mask。这使得选择逻辑易于检查，但无法提供融合内核的计算节省。

下图总结了流程。Lightning 索引器对候选 token 进行评分，选择器保留前 K 个位置，生成的 mask 限制通常的注意力 softmax。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek 稀疏注意力流程图" width="700px" />

*图 3. DSA 首先对候选 token 进行评分，然后为最终注意力 mask 保留前 K 个 token。*

&nbsp;
## 实现

`gpt_with_kv_dsa.py` 提供了：

| 类 | 描述 |
|---|---|
| `LightningIndexer` | 用于历史 token 相关性的轻量级多头评分器。 |
| `MultiHeadAttentionWithDSA` | 标准 MHA 带有 DSA 稀疏掩码 + 可选 KV 缓存。 |
| `GPTModel` | 替换为 `MultiHeadAttentionWithDSA` 的 GPT 风格模型。 |

本实现遵循本存储库中其他补充材料的风格，可作为独立脚本运行。它旨在使 DSA 机制在小型 GPT 风格模型中可检查。它未实现 DeepSeek 的完整 MLA 栈、融合稀疏内核或部署特定的优化。

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
| `--index_n_heads` | 4 | 轻量级索引器头的数量 (H_I)。 |
| `--index_head_dim` | 64 | 每个索引器头的维度。 |
| `--topk` | 64 | 每个查询关注的 token 数量 (k)。短序列时上限为序列长度。 |

&nbsp;
## 与 DeepSeek V3.2 的关系

全规模的 DeepSeek-V3.2 模型使用多头潜在注意力 (MLA，参见 [../05_mla](../05_mla)) 以及 DSA，并且索引器查询是从共享的压缩潜在表示派生的，而不是原始输入。DeepSeek-V3.2 使用与 DeepSeek-V3.2-Exp 相同的架构，DSA 首次在其中引入和测试。

这里复现了关键的选择思想。一个廉价的学习点积评分器在注意力 softmax 之前将每个查询限制为最相关的 token。

下面报告的推理成本比较对于理解 DSA 在长上下文部署中的重要性很有用。节省取决于生产内核和服务基础设施，因此不应将此图视为本文件夹中教学实现的基准。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="DeepSeek 稀疏注意力推理成本比较" width="800px" />

*图 4. DeepSeek 报告的 DSA 在长上下文服务中的推理成本节省，来自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

&nbsp;
## 参考文献

- DeepSeek V3.2 技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp 模型卡与参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的 "从 DeepSeek V3 到 V3.2：架构、稀疏注意力与 RL 更新"：https://magazine.sebastianraschka.com/p/technical-deepseek
