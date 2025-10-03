# 附加材料：KV Cache

**这个文件夹实现了在 GPT 模型中添加 KV 缓存。**

&nbsp;
## 概述
简而言之，KV缓存存储中间键（K）和值（V）计算结果以供推理时重用，这在生成响应时能显著提升速度。缺点是它增加了代码的复杂性，提高了内存使用，且在训练过程中无法使用。然而，在部署LLMs时，推理速度的提升往往足以抵消代码复杂性和内存增加带来的代价。

&nbsp;
## 如何生效

想象LLM正在生成文本。具体来说，假设LLM收到了以下提示："Time flies"。

下图展示了使用修改自第3章的图来突出显示键向量和值向量的底层注意力分数计算的一个示例：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如在第2章和第4章中所学，LLMs每次生成一个词（或标记）。假设LLM生成了单词"fast"，那么下一轮的提示就变成了"Time flies fast"。这在下面的下一个图中进行了说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

如图所示，通过比较前两个图，可以发现前两个token的keys和value向量完全相同，在每次下一个token的文本生成轮次中重新计算它们是浪费的。

因此，KV缓存的思路是实现一种缓存机制，存储先前生成的键和值向量以便重用，这有助于避免不必要的重复计算。

&nbsp;
## KV Cache实现

实现KV缓存的方法有很多，主要思路是在每一步生成过程中，只计算新生成token的键和值张量。

在此选择一个简单的实现，它强调代码的可读性。直接滚动查看代码变更就能最直观地了解其实现方式。

本路径下中有两个文件：
1. [`gpt_ch04.py`](gpt_ch04.py)：从第3章和第4章中提取的独立代码，用于实现LLM并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上述相同，但已进行必要的更改以实现KV缓存

可以：

a. 打开[`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)文件，查找标记新更改的 # NEW 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 使用文件差异工具查看两个代码文件，以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

总结实现细节，以下是一个简短的说明。

&nbsp;
### 1.注册Cache缓冲区

在`MultiHeadAttention`构造函数内部，添加两个非持久性缓冲区`cache_k`和`cache_v`，它们将跨步骤存储连接的键和值：

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
```

&nbsp;
### 2.带有`use_cache`标志的前向传播

接下来，扩展`MultiHeadAttention`类的`forward`方法以接受`use_cache`参数。在将新的`token`块投影到`keys_new`、`values_new`和`queries`后，初始化`kv`缓存或向缓存中追加：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;
### 3.清除缓存

在生成文本时，在不同的独立序列之间（例如文本生成调用之间），必须重置两个缓冲区，因此也在`MultiHeadAttention`类中添加了一个缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;
### 4.在完整模型中传播`use_cache`

在`MultiHeadAttention`类修改完成后，现在修改`GPTModel`类。首先，添加一个用于跟踪标记索引的位置：

```python
self.current_pos = 0
```

然后，将这个单行块调用替换为显式的循环，将`use_cache`传递给每个`Transformer`块：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还要求对`TransformerBlock`类进行微小修改以接受`use_cache`参数：

```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，向`GPTModel`添加一个模型级别的重置，以便一次性清除所有块缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;
### 5.生成过程中使用KV缓存

通过修改`GPTModel`、`TransformerBlock`和`MultiHeadAttention`，最终，以下在简单的文本生成函数中使用`KV`缓存的用法：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，在 c)中仅通过`logits = model(next_idx, use_cache=True)`向模型提供新`token`。如果没有缓存，将整个输入`logits = model(idx[:, -ctx_len:], use_cache=False)`提供给模型，因为它没有可重用的存储键和值。

&nbsp;
## 简单的性能比较
在概念层面上介绍了KV缓存后，关键问题在于它在小示例上的实际表现如何。为了尝试实现，可以将上述两个代码文件作为Python脚本运行，这将运行一个参数量为124M的小型LLM，以生成200个新标记（以"Hello, I am"为4标记的提示开始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在一台搭载M4芯片（CPU）的Mac Mini上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，如上所示，使用一个参数量只有124M的小模型和长度为200个tokens的短序列，已经获得了约5倍的加速效果。（注意，这个实现是为了代码可读性而优化的，而不是为了CUDA或MPS运行时速度而优化的，后者需要预先分配张量，而不是重新创建和连接它们。）

**注意**：在两种情况下，模型都会生成“胡言乱语”，即看起来像这样的文本：

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为还没有训练模型。下一章将训练模型，可以使用训练好的模型的KV缓存（但是KV缓存仅用于推理时）来生成连贯的文本。这里使用未训练的模型以保持代码简单。

然而，更重要的是，`gpt_ch04.py`和`gpt_with_kv_cache.py`的实现都生成了完全相同的文本。这表明KV缓存实现正确——索引错误很容易导致结果出现分歧。

&nbsp;
## KV缓存的优势和劣势

随着序列长度的增加，KV缓存的优点和缺点在以下方面变得更加明显：

- [优势]计算效率提高：如果没有缓存，步骤 *t* 的注意力必须将新的查询与 *t* 个先前的键进行比较，因此累积工作量按平方级增长，为O(n²)。有了缓存，每个键和值只需计算一次，然后重复使用，将每步的总体复杂度降低到线性级，为O(n)

- [劣势]内存使用量线性增长：每个新token都会追加到KV缓存中。对于长序列和大型的LLMs，累积的KV缓存会变得更大，这可能会消耗大量甚至过高的（GPU）内存。作为解决方案，可以截断KV缓存，但这会增加更多的复杂性（但再次强调，在部署LLMs时，这很可能还是值得的。）

&nbsp;
## 优化KV缓存实现

虽然上述的概念性KV缓存实现有助于清晰理解，并且主要面向代码可读性和教育目的，但在实际场景中部署（尤其是使用较大模型和较长的序列长度时）需要更仔细的优化。

&nbsp;
### 扩展缓存时常见的陷阱

- **内存碎片化和重复分配**：通过前面所示的方式，通过`torch.cat`连续拼接张量，会导致频繁的内存分配和重新分配，从而造成性能瓶颈。

- **内存使用呈线性增长**：如果没有适当的处理，KV缓存的大小对于非常长的序列来说会变得不切实际。

&nbsp;
#### Tip 1：预分配内存

与其反复拼接张量，可以根据预期的最大序列长度预分配一个足够大的张量。这可以确保内存使用的一致性并减少开销。在伪代码中，这可能看起来像这样：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理过程中，可以直接将这些预分配张量的切片进行写入。

#### Tip 2：通过滑动窗口截断缓存

为了避免GPU内存爆满，可以实现一个带有动态截断的滑动窗口方法。通过滑动窗口，仅保留缓存中的最后window_size个token：

```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实践中的优化

可以在[`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py)文件中找到这些优化；因为滑动窗口的设置，注意力计算操作只会窗口中的tokens间进行。

在一台搭载M4芯片（CPU）的Mac Mini上，使用200个tokens的生成和与上下文长度相等的窗口大小（为确保结果一致），代码运行时间对比如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，对于这个微小的模型，在CUDA设备上的速度优势消失了，因为设备传输和通信超过了KV缓存对这个小模型带来的好处。

&nbsp;
## 额外资源

1. [Qwen3 from-scratch KV cache benchmarks](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 from-scratch KV cache benchmarks](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 本README的更详细说明