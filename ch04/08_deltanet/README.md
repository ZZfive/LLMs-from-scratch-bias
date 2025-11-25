# 线性注意力中的Gated DeltaNet

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)和[Kimi Linear](https://arxiv.org/abs/2510.26692)提出了混合式Transformer，将注意力机制替换为在上下文长度上呈线性（而非二次）扩展的变体。

Qwen3-Next与Kimi Linear都采用3:1的层级比例，也就是每3个使用线性Gated DeltaNet变体的Transformer模块之后，会接一个使用完整注意力的模块，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



&nbsp;

## 引言与概览

Gated DeltaNet是受循环神经网络启发的线性注意力变体，其中包括来自论文[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)的门控机制。从某种意义上说，Gated DeltaNet是加入了Mamba式门控的DeltaNet，而DeltaNet本身就是一种线性注意力机制。

Kimi Linear通过Kimi Delta Attention（KDA）机制改进了Qwen3-Next的线性注意力，本质上是对Gated DeltaNet的细化。Qwen3-Next使用标量门（每个注意力头一个值）来控制记忆衰减速率，而Kimi Linear将其替换为按通道的门控（针对每个特征维度）。作者指出，这能对记忆进行更精细的控制，从而提升长上下文推理能力。

此外，在完整注意力层中，Kimi Linear用多头潜在注意力（MLA）替换了Qwen3-Next的门控注意力层（实质上是带输出门控的普通多头注意力）。这与在DeepSeek V3/R1部分讨论过的MLA相同，但额外加入了一个门。（回顾一下，MLA会压缩键/值空间以减小KV缓存。）

Kimi Linear中的MLA暂未加入门控，作者希望便于与标准MLA做直接对比。但他们[表示](https://x.com/yzhang_cs/status/1984631714464088563)计划未来加入该门控。

由于已经在[../05_mla](../05_mla)中实现了MLA，本额外材料主要聚焦于Gated DeltaNet。


&nbsp;

## 门控注意力

在介绍Gated DeltaNet之前，先简单讨论一下门控。上一张图的Qwen3-Next架构上半部分展示了“门控注意力”，它实际上是在常规完整注意力基础上增加了一个sigmoid门。

下面示例展示了在第3章`MultiHeadAttention`代码上额外添加门控的方式：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate        
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



可以看到，在正常的注意力计算之后，模型会根据同一输入额外生成门控信号，使用sigmoid将其限制在0到1之间，并与注意力输出相乘。这样模型就可以动态放大或缩小特定特征。Qwen3-Next的开发者[提到](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)这有助于训练稳定性：

> ……注意力输出的门控机制可以消除Attention Sink和Massive Activation等问题，确保整个模型的数值稳定性。


&nbsp;

## Gated DeltaNet

那么，Gated DeltaNet是什么？Gated DeltaNet（Gated Delta Network）是Qwen3-Next的线性注意力层，用来替代标准的softmax注意力，它正是引自前文提到的[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)论文。

Gated DeltaNet最初作为Mamba2的改进版本提出，将Mamba2的门控衰减机制与delta规则结合在一起。

Mamba是一种状态空间模型（state-space model），是可替代Transformer的大型主题，未来可以单独讨论。

这里的delta规则指的是计算新旧值之间的差（delta，Δ），并据此更新一个作为记忆的隐藏状态（后面会详细介绍）。

（题外话：熟悉经典机器学习文献的读者可以将其类比为受生物启发的Hebbian学习——“一起激活的细胞，会彼此连接”。它基本上是感知器更新和基于梯度下降学习的前身，不过是无监督方式。）

Gated DeltaNet中的门控与上文的门控注意力类似，但使用的是SiLU而非sigmoid，如下图所示（选择SiLU可能是为了改进梯度流与稳定性）。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

不过如图所示，Gated DeltaNet中的“gated”还包含其他几种门：

- `α`（衰减门）控制记忆随时间衰减或重置的速度；
- `β`（更新门）控制新输入对状态的更新幅度。

在代码层面，如下示例演示了一个去掉卷积混合的简化版本（代码参考[Qwen3 团队的官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)）：

```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)
        
        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation
  
        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)
        gate = gate.transpose(1, 2)  # NEW

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t.exp()
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（为简洁起见，省略了Qwen3-Next与Kimi Linear中用于混合局部信息的卷积，便于聚焦循环部分并保持可读性。）

如上所示，它与标准（或门控）注意力有许多不同。

在门控注意力中，模型还是像往常一样计算token间的注意力（每个token都会关注其他所有token）。之后再通过一个sigmoid门控决定保留多少输出。归根结底，它还是随着上下文长度呈二次增长的缩放点积注意力。

回顾一下，缩放点积注意力的公式是 softmax(QKᵀ)V，其中Q和K是*n*×*d*的矩阵；*n*为token数，*d*为嵌入维度。因此QKᵀ会得到一个*n*×*n*的注意力矩阵，再与*n*×*d*的V相乘：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

而在Gated DeltaNet中，不再存在*n*×*n*的注意力矩阵。模型改为逐token处理，维护一个随新token不断更新的记忆状态`S`。它的实现如下（`S`在每个时间步*t*均会被递归更新）：

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t.exp()
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

这里的不同门控用于调节记忆状态的变化：

- α（`alpha`）控制旧记忆遗忘（衰减）的程度；
- β（`beta`）控制当前时间步*t*的token对记忆的更新幅度；
- 还有一个输出门控（上述代码片段未展示），类似门控注意力，用于决定最终输出保留多少。

因此，从某种程度上讲，Gated DeltaNet的状态更新神似RNN。优势在于它以线性（for 循环）而非二次复杂度扩展上下文长度。

劣势则是在放弃完整成对注意力后，整体上下文建模能力有所下降。

Gated DeltaNet在一定程度上仍能捕捉上下文，但必须通过固定大小的记忆`S`来传递。这个记忆高效但容量有限，与RNN将既往信息压缩到隐藏状态的方式类似。

这也是Qwen3-Next和Kimi Linear没有完全用DeltaNet取代所有注意力层，而是采用前述3:1比例的原因。

&nbsp;

## DeltaNet的内存节省

上一节提到，相较完整注意力，DeltaNet在上下文长度维度上的计算复杂度是线性的而不是二次的。

除了线性计算复杂度，DeltaNet的另一个优势是内存节省，因为DeltaNet模块不会扩大KV缓存（有关KV缓存，可参见[../03_kv-cache](../03_kv-cache)）。它只需要维护固定大小的循环状态，因此内存不会随上下文长度增长。

对于常规多头注意力（MHA），KV缓存大小可按以下方式估算：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（乘以2是因为缓存中既要存键又要存值。）

而对于上面实现的简化版DeltaNet：

```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

可以看到，`KV_cache_DeltaNet` 的公式中不再含有上下文长度`n_tokens`。此外，只需存储状态`S`，而无需单独的键和值，所以`2 × bytes`变成了`bytes`。不过也注意这里出现了`d_head × d_head`的二次项，它来自以下状态定义：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

由于head维度通常较小（例如Qwen3-Next中为128），这个二次项一般问题不大。

完整版（带卷积混合）的公式会更复杂，需要考虑卷积核大小等因素，但上述公式足以体现Gated DeltaNet的主要趋势和动机。

可以运行下面的辅助脚本来可视化不同上下文长度下的内存估计与节省情况：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

需要注意，这里`head_dim = emb_dim / n_heads`，即 2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>
