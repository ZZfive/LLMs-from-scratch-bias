# 从零构建 Qwen3

本文件夹中的 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter notebook 包含了 Qwen3 0.6B、1.7B、4B、8B 和 32B 的从零实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

本文件夹中的 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) 和 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) Jupyter notebook 包含了 30B-A3B 专家混合（MoE）模型的从零实现，涵盖思维（Thinking）、指令（Instruct）和编程（Coder）变体。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# Qwen3 从零实现代码

此文件夹中的独立 notebook 按线性方式组织了从零实现代码：

1. [standalone-qwen3.ipynb](standalone-qwen3.ipynb)：基础版的稠密 Qwen3 模型实现
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb)：与上面相同，但加入了 KV 缓存以提高推理效率
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb)：与第一个 notebook 类似，但实现的是专家混合（MoE）变体
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb)：与上面相同，但加入了 KV 缓存以提高推理效率

此外，代码也被整理成了 Python 包，位于[这里](../../pkg/llms_from_scratch/)（包含单元测试和 CI），可以按下文所述方式运行。

&nbsp;
# 训练

`Qwen3Model` 类的实现风格与 `GPTModel` 类似，因此可以作为第 5 章训练以及第 6、7 章微调代码中的即插即用替代方案。

&nbsp;
# 通过`llms-from-scratch`包使用Qwen3

为了更方便地使用 Qwen3 的从零实现，你也可以使用基于本仓库源码构建的 `llms-from-scratch` PyPI 包，位于 [pkg/llms_from_scratch](../../pkg/llms_from_scratch)。

&nbsp;
#### 1) 安装

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) 模型和文本生成设置

指定要使用的模型：

```python
USE_REASONING_MODEL = True
# 如果 USE_REASONING_MODEL = False，则使用基础模型

USE_INSTRUCT_MODEL = False
# 如果
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# 则使用指令模式（不带思维链）
# 如果 USE_REASONING_MODEL = False，则此设置无效


# 使用
# USE_REASONING_MODEL = True
# 也适用于Qwen3 Coder Flash模型
```

下面是用户可自定义的基础文本生成设置。以生成 150 个 token 为例，0.6B 模型大约需要 1.5 GB 内存。

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a) 0.6B模型的权重下载和加载

下面的代码会根据上面选择的模型类型（思维或基础）自动下载权重文件。请注意，本节重点关注 0.6B 模型。如果你想使用更大的模型（1.7B、4B、8B 或 32B），请跳过本节并继续阅读第 3b) 节。

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

然后按如下方式加载模型权重：

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b) 更大Qwen模型的权重下载和加载

如果你想使用更大的 Qwen 模型，例如 1.7B、4B、8B 或 32B，请使用下面这段代码，而不是 3a) 中的代码；这需要额外安装一些依赖：

```bash
pip install safetensors huggingface_hub
```

然后使用以下代码（按需修改 `USE_MODEL` 以选择想要的模型规模）

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

现在，将权重下载并加载到 `model` 中：

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # 仅对MoE模型需要
del weights_dict  # 删除权重字典以释放内存
```


&nbsp;

#### 4) 初始化分词器

下面的代码会下载并初始化分词器：

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```



&nbsp;

#### 5) 生成文本

最后，可以使用下面的代码生成文本：

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```






```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

使用 Qwen3 0.6B 思维模型时，输出大致应与下面类似（示例在 A100 上运行）：

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



对于更大的模型，你可能更偏好流式生成版本，它会在生成过程中立即打印每个 token：

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



&nbsp;

#### 专业技巧1：通过编译加速推理

为了获得最高可达 4 倍的加速，可以将

```python
model.to(device)
```

替换为

```python
model.to(device)
model = torch.compile(model)
```

注意：编译会带来显著的前期开销，通常需要几分钟，并且加速效果要在第一次 `generate` 调用之后才会体现出来。

下表展示了在 A100 上连续调用 `generate` 时的性能对比：

|                          | 硬件          | Tokens/sec | 内存   |
| ------------------------ | ------------- | ---------- | ------ |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B 编译版 | Nvidia A100 GPU | 107        | 1.99 GB  |


&nbsp;
#### 专业技巧2：通过KV缓存加速推理

在 CPU 上运行模型时，可以使用支持 KV 缓存的 `Qwen3Model` 作为即插即用替代，以显著提升推理性能。（有关 KV 缓存的更多信息，请参阅我的文章[从零理解和实现 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。）

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

请注意，峰值内存使用量这里只列出了 Nvidia CUDA 设备上的数值，因为这类设备更容易精确测量。不过，其他设备上的内存占用通常也会比较接近，因为它们使用了类似的精度格式；同时，KV 缓存版本在生成 150 个 token 时的内存占用甚至更低。当然，不同设备的矩阵乘法实现方式可能不同，因此峰值内存需求也可能有所差异；另外，随着上下文长度增长，KV 缓存本身的内存占用也会明显增加。

| 模型           | 模式              | 硬件          | Tokens/sec | GPU内存（VRAM） |
| --------------- | ----------------- | ------------- | ---------- | --------------- |
| Qwen3Model 0.6B | 常规版            | Mac Mini M4 CPU | 1          | -               |
| Qwen3Model 0.6B | 常规版编译        | Mac Mini M4 CPU | 1          | -               |
| Qwen3Model 0.6B | KV缓存            | Mac Mini M4 CPU | 80         | -               |
| Qwen3Model 0.6B | KV缓存编译        | Mac Mini M4 CPU | 137        | -               |
|                 |                   |               |            |                 |
| Qwen3Model 0.6B | 常规版            | Mac Mini M4 GPU | 21         | -               |
| Qwen3Model 0.6B | 常规版编译        | Mac Mini M4 GPU | 错误        | -               |
| Qwen3Model 0.6B | KV缓存            | Mac Mini M4 GPU | 28         | -               |
| Qwen3Model 0.6B | KV缓存编译        | Mac Mini M4 GPU | 错误        | -               |
|                 |                   |               |            |                 |
| Qwen3Model 0.6B | 常规版            | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model 0.6B | 常规版编译        | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model 0.6B | KV缓存            | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model 0.6B | KV缓存编译        | Nvidia A100 GPU | 90         | 1.48 GB           |

请注意，上述所有设置都经过测试，并且会生成相同的文本输出。


&nbsp;

#### 专业技巧3：批处理推理

还可以通过批量推理进一步提高吞吐量。虽然这并不是完全可直接比较的设置，因为此时会同时处理更多输入序列，从而提高每秒 token 吞吐量，但代价是更高的内存占用。

这只需要对提示准备逻辑做少量修改。例如，考虑下面这个批量提示示例：

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 缓存版本的代码与之类似，只是需要使用下面这些即插即用替换：

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


下面的实验使用批大小 8 运行。

| 模型            | 模式              | 硬件          | 批大小 | Tokens/sec | GPU内存（VRAM） |
| ---------------- | ----------------- | ------------- | ------ | ---------- | --------------- |
| Qwen3Model  0.6B | 常规版            | Mac Mini M4 CPU | 8      | 2          | -               |
| Qwen3Model 0.6B  | 常规版编译        | Mac Mini M4 CPU | 8      | -          | -               |
| Qwen3Model 0.6B  | KV缓存            | Mac Mini M4 CPU | 8      | 92         | -               |
| Qwen3Model 0.6B  | KV缓存编译        | Mac Mini M4 CPU | 8      | 128        | -               |
|                  |                   |               |        |            |                 |
| Qwen3Model 0.6B  | 常规版            | Mac Mini M4 GPU | 8      | 36         | -               |
| Qwen3Model 0.6B  | 常规版编译        | Mac Mini M4 GPU | 8      | -          | -               |
| Qwen3Model 0.6B  | KV缓存            | Mac Mini M4 GPU | 8      | 61         | -               |
| Qwen3Model 0.6B  | KV缓存编译        | Mac Mini M4 GPU | 8      | -          | -               |
|                  |                   |               |        |            |                 |
| Qwen3Model 0.6B  | 常规版            | Nvidia A100 GPU | 8      | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | 常规版编译        | Nvidia A100 GPU | 8      | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV缓存            | Nvidia A100 GPU | 8      | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV缓存编译        | Nvidia A100 GPU | 8      | 280        | 1.75 GB           |
