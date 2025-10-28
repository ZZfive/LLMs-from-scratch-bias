# 将GPT转换为Llama

这个文件夹包含将第4章和第5章的GPT实现转换为Meta AI的Llama架构的代码，推荐按以下顺序阅读：

- [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb)：包含逐步将GPT转换为Llama 2 7B的代码，并从Meta AI加载预训练权重
- [converting-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb)：包含将Llama 2模型转换为Llama 3、Llama 3.1和Llama 3.2的代码
- [standalone-llama32.ipynb](standalone-llama32.ipynb)：一个独立的笔记本，实现 Llama 3.2

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">

&nbsp;
## 通过llms-from-scratch包使用Llama 3.2

要轻松使用Llama 3.2 1B和3B模型，也可以使用基于此仓库源代码在[pkg/llms_from_scratch](../../pkg/llms_from_scratch)中的 llms-from-scratch PyPI包。

&nbsp;
### 1）安装

```bash
pip install llms_from_scratch blobfile
```

&nbsp;
### 2）模型和文本生成设置

指定具体使用的模型：

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

用户可以定义的基本文本生成设置。请注意，推荐的8192个token的上下文大小需要大约3GB的VRAM来进行文本生成示例。

```python
# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
### 3）权重下载和加载

根据上面选择的模型自动下载权重文件：

```python
import os
import requests

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {MODEL_FILE}")
```

模型权重随后按以下方式加载：

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
### 4）初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

&nbsp;
### 5）生成文本

最后，基于以下代码生成文本

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
```

在使用Llama 3.2 1B Instruct模型时，输出应与下方所示相似：

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

&nbsp;
### 技巧1：使用FlashAttention加速推理

可以使用`Llama3ModelFast`代替`Llama3Model`。如需更多信息，建议查看[pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py)代码。

`Llama3ModelFast`将`GroupedQueryAttention`模块中手动实现的点积代码替换为PyTorch的`scaled_dot_product`函数，该函数在Ampere显卡或更新的显卡上使用`FlashAttention`。

下表展示了在A100上的性能比较：

|                 | Tokens/sec | Memory  |
| --------------- | ---------- | ------- |
| Llama3Model     | 42         | 2.91 GB |
| Llama3ModelFast | 54         | 2.91 GB |

&nbsp;
### 技巧2：通过编译加速推理

进行以下替换，实现高达4倍的加速：

```python
model.to(device)
```

```python
model = torch.compile(model)
model.to(device)
```

注意：编译时存在显著的持续多分钟的前期成本，加速效果在第一次`generate`调用后才会生效。

下表展示了在A100上连续`generate`调用时的性能比较：

|                 | Tokens/sec | Memory  |
| --------------- | ---------- | ------- |
| Llama3Model     | 170        | 3.12 GB |
| Llama3ModelFast | 177        | 3.61 GB |

&nbsp;
### 技巧3：使用KV Cache加速

在CPU上运行模型时，可以使用KV缓存`Llama3Model`作为即插即用的替代方案来显著提升推理性能。有关KV缓存的信息，请参阅[《从零开始理解和编写LLMs中的KV缓存》](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)文章。

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

请注意，峰值内存使用量仅针对Nvidia CUD设备列出，因为计算起来更简单。然而，其他设备的内存使用情况可能相似，因为它们使用相似的精度格式，并且 KV缓存存储导致生成的150个tokens文本的内存使用更低（但是，不同的设备可能在矩阵乘法方面有不同的实现，从而导致不同的峰值内存需求；对于更长的上下文长度，KV缓存内存可能会大幅增加）。

| Model       | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ----------- | ----------------- | --------------- | ---------- | ----------------- |
| Llama3Model | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | KV cache          | Mac Mini M4 CPU | 68         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 CPU | 86         | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Mac Mini M4 GPU | 15         | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Llama3Model | KV cache          | Mac Mini M4 GPU | 62         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Nvidia A100 GPU | 42         | 2.91 GB           |
| Llama3Model | Regular compiled  | Nvidia A100 GPU | 170        | 3.12 GB           |
| Llama3Model | KV cache          | Nvidia A100 GPU | 58         | 2.87 GB           |
| Llama3Model | KV cache compiled | Nvidia A100 GPU | 161        | 3.61 GB           |

请注意，上述所有设置均已测试以确保生成相同的文本输出。