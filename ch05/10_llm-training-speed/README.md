# 用于更快LLM训练的PyTorch性能优化技巧

注意本书的编写目的是教育，这意味着原始代码故意保持简单。这是为了提高可读性并确保在不同硬件（包括CPU和GPU）上的兼容性。不过，您可能对一些更高级的PyTorch和GPU功能感到好奇，以使LLM训练更具性能。

本文件夹包含三个代码文件，展示了第5章介绍的LLM和训练函数的性能优化：

1. [`00_orig.py`](00_orig.py)：用于CPU和单GPU训练的原始第5章代码。  
   ➤ 运行方式：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：单GPU训练的优化版本。  
   ➤ 运行方式：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多GPU训练优化版本。  
   ➤ 运行方式：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   （**注意：** 为保持与`01_opt_single_gpu.py`相比的最小变化，此脚本仅支持通过上面的`torchrun`进行多进程处理。这意味着**不**支持通过`python 02_opt_multi_gpu_ddp.py`进行多GPU支持）

**请注意，这些优化将训练速度从12,525 tokens/秒（单个A100）提升到142,156 tokens/秒（单个A100）和419,259 tokens/秒（4个A100）。**

我计划在未来某个时候更详细地阐述这些差异。目前，查看代码改进的最简单方法是在Visual Studio Code中打开文件并通过"比较所选内容"功能查看差异。

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


&nbsp;
## 单GPU速度比较

如上所述，我计划在未来更详细地阐述这些更改。目前，此部分包含每个修改以tokens/秒为单位的简单性能概览。所有实验均在A100 GPU上运行。

&nbsp;
### 基线

请注意，`00_orig.py`作为基线，不包含重大修改，除了以下内容外，使用第5章的代码：

- 4倍的上下文长度（这解释了`00_orig.py`相比第5章的相对较大内存占用）；
- 4倍批大小变化（另一个导致`00_orig.py`相对较大内存占用的原因）；
- 更大的公共领域书籍以增加训练数据大小。

超参数不是为了最小化损失和减少过拟合而优化的，LLM在最后生成的文本可能不会非常复杂；但是，这不应该有问题，因为主要要点是`tok/sec`指标，用作速度参考（越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

请注意，`01_opt_single_gpu.py`包含下面顺序列出的所有修改。

比较始终基于上一节第一个epoch之后的平均tok/sec和已分配内存。

&nbsp;
### 1. 动态创建因果掩码

- 不是保存因果掩码，而是动态创建因果掩码以减少内存使用（这里影响最小，但在支持131k输入tokens的长上下文大小模型（如Llama 3.2）中可能会累积）

之前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

之后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用张量核心

- 使用张量核心（仅适用于A100等Ampere GPU及更新版本）

之前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 融合AdamW优化器

- 通过设置`fused=True`为`AdamW`使用融合内核

之前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 数据加载器中的固定内存

- 在数据加载器中使用`pin_memory=True`来预分配和重用GPU内存

之前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用bfloat16精度

- 从32位浮点数切换到16位脑浮点数（bfloat16）精度（有关此主题的更多信息，请参阅[我的文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

之前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 用PyTorch类替换从零开始的代码

- 用PyTorch的原生实现替换从零开始的LayerNorm和GeLU实现

之前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

之后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用FlashAttention

- 使用PyTorch的自注意力函数和FlashAttention，而不是我们的从零开始的多头注意力实现。

之前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

之后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用`pytorch.compile`

- 使用`torch.compile(model)`。请注意，在加速之前第一次迭代总是较慢。由于`Avg tok/sec`测量仅包含平均计算的第一行，我们现在使用第1个epoch结束时的`Step tok/sec`。

之前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

之后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows注意事项**

- 在Windows上编译可能比较棘手
- `torch.compile()`使用Inductor，它JIT编译内核并需要有效的C/C++工具链
- 对于CUDA，Inductor还依赖Triton，可通过社区包`triton-windows`获得
  - 如果您看到`cl not found`，[安装带有"C++工作负载"的Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)并从"x64本机工具"提示运行Python
  - 如果您看到`triton not found`且使用CUDA，安装`triton-windows`（例如，`uv pip install "triton-windows<3.4"`）。
- 对于CPU，读者进一步推荐遵循此[Windows的PyTorch Inductor指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
  - 这里，重要的是在安装Visual Studio 2022时安装英语语言包以避免UTF-8错误
  - 此外，请注意代码需要通过"Visual Studio 2022开发人员命令提示符"运行，而不是通过笔记本运行
- 如果此设置证明棘手，您可以跳过编译；**编译是可选的，所有代码示例不使用它也能正常工作**

---

&nbsp;
### 9. 词汇表填充

- 这里，我们将词汇表大小从50,257略微增加到50,304，这是64的最近倍数。这个技巧是我的前同事Carlos Mocholi建议我的，他提到这最初来自Andrej Karpathy（可能来自[这个帖子](https://x.com/karpathy/status/1621578354024677377)）。Karpathy的建议基于与PyTorch团队的互动，他们对`torch.compile`给出了建议，如[Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29)所提及。对此的一个好资源是[NVIDIA关于张量形状的指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，其中批大小和线性层维度通常选择为某些值的倍数。此外，NVIDIA的Megatron团队很久以前就描述了词汇表填充技巧（参见2019年[Megatron-LM:使用模型并行训练多十亿参数语言模型](https://arxiv.org/abs/1909.08053)论文）。

之前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

之后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增加批大小

- 最后，我们将批大小增加到GPU支持的最大2的幂

之前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

之后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


&nbsp;
## 多GPU速度比较

这可能不是一个完全公平的比较，因为我们现在使用4个GPU而不是1个，但使用分布式数据并行（如果训练不受有限GPU内存瓶颈限制，这是可用的最快多GPU技术），当然可以带来明显的加速：

之前（单GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

之后（4个GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
