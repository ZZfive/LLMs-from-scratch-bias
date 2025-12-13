# 额外的分类微调实验

下表添加了实验来回答有关各种设计选择的额外问题。第一行使用与主章节相同的设置作为参考。
例如，

- 比较第1行和第2行回答问题："当训练最后一个或第一个token时，性能差异是什么？"；
- 比较第1行和第3行回答问题："当只训练最后一层而不是最后一个块时，性能差异是什么？"；
- 依此类推。

&nbsp;

|      | Model              | Weights    | Trainable token position | Trainable layers | Context length                                         | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | pretrained | first                    | last_block       | longest train ex. (120)                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | pretrained | last                     | last_layer       | longest train ex. (120)                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | pretrained | last                     | last_two_blocks  | longest train ex. (120)                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | pretrained | last                     | all              | longest train ex. (120)                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | pretrained | last                     | last_block       | longest train ex. (120)                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | pretrained | last                     | all              | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | random     | last                     | all              | longest train ex. (120)                                | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | pretrained | last                     | last_block       | context length (1024)                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 1)                    | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 8)                    | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | pretrained | last                     | last_block       | flexible (last non-padding position)                   | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120); but no causal mask            | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120) and `ignore_index` for padding | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | pretrained | last + pooled embeddings | last_block       | longest train ex. (120)                                | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;

### 使用方法

可以使用以下代码来复现实验：

- 第1行：`python additional_experiments.py`
- 第2行：`python additional_experiments.py --trainable_token_pos first`
- 第3行：`python additional_experiments.py --trainable_layers last_layer`
- 第4行：`python additional_experiments.py --trainable_layers last_two_blocks`
- 第5行：`python additional_experiments.py --trainable_layers all`
- 第6行：`python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 第7行：`python additional_experiments.py --model_size "gpt2-large (774M)"`
- 第8行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 第9行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- 第10行：`python additional_experiments.py --weights random --trainable_layers all`
- 第11行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 第12行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 第13行：`python additional_experiments.py --context_length "model_context_length"`
- 第14行：`python additional_experiments.py --no_padding --batch_size 1`
- 第15行：`python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 第16行：`python additional_experiments.py --trainable_token_pos "flexible"`
- 第17行：`python additional_experiments.py --disable_causal_mask`
- 第18行：`python additional_experiments.py --ignore_index 50256`
- 第19行：`python additional_experiments.py --average_embeddings`

原作者故意保持LLM和数据集较小，这样就可以在普通的笔记本电脑（如MacBook Air M3）上运行训练，大约15分钟（默认设置），如果没有GPU的话。

&nbsp;

### 解释

1. **训练最后一个vs第一个输出token位置（第1行vs第2行）**：训练最后一个输出token位置比第一个的性能显著更好。由于因果自注意力掩码，这种改进是预期的。
2. **训练最后一个Transformer块vs最后一层（第1行vs第3行）**：训练整个最后一个Transformer块也比只训练最后一层的结果显著更好。
3. **训练最后一个vs最后两个Transformer块（第1行vs第4行）**：训练最后两个Transformer块而不是只训练最后一个块，可带来明显的3.33%准确率提升。
4. **训练最后一个Transformer块vs所有层（第1行vs第5行）**：训练所有层比只训练最后一个Transformer块有 modest ~2%的改进，但训练时间几乎长三倍。而且，它的表现也不如只训练12个Transformer块中的最后两个。
5. **使用更大的预训练模型（第1行vs第6行，以及第1行vs第7行和第8行）**：使用3倍大的预训练模型会导致更差的结果。然而，使用5倍大的模型相比初始模型提高了性能，这是预期的。类似地，12倍大的模型进一步改善了预测性能。（中型模型可能预训练得不够好，或者特定的微调配置对这个模型效果不太好。）
6. **使用随机权重vs预训练权重的模型（第1行和第5行vs第10行）**：使用随机权重的模型结果仅比使用预训练权重稍差（差3%和1.3%）。
7. **使用LoRA（低秩适应）vs训练所有层（第11行vs第5行，第12行vs第9行）**：保持模型冻结并添加可训练的LoRA层（详见[附录E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练所有模型参数的可行的替代方案，甚至提高了1个百分点的性能（第11行vs第5行）。从使用LoRA时训练和验证准确率之间约低1%的差距可以看出，这可能是由于过拟合减少。此外，使用LoRA也更节省内存，因为需要更新的参数更少。当训练更大的模型时（第12行vs第9行），也可以看到LoRA训练得更快（5.79分钟而不是8.12分钟）。
8. **填充输入到完整上下文长度vs最长训练样本（第1行vs第13行）**：填充输入到完整支持的上下文长度的结果显著更差。
9. **填充vs无填充（第1行vs第14和15行，以及第16行）**：`--no_padding`选项禁用数据集中的填充，这需要使用批量大小1来训练模型，因为输入长度可变。这带来了更好的测试准确率，但训练时间更长。在第15行中，另外启用梯度累积8步，以达到与其他实验相同的批量大小，这有助于减少过拟合并略微提升测试集准确率。在第16行中，应用了填充，但token位置是基于最后一个非填充token选择的。第16行在数学上应该与使用梯度累积的第15行相似。然而，由于在token数量不等的情况下梯度累积的一些挑战，可能存在小的差异（这在[this](https://unsloth.ai/blog/gradient)博客文章中讨论过）。
10. **禁用因果注意力掩码（第1行vs第17行）**：禁用多头注意力模块中使用的因果注意力掩码。这意味着所有token都可以关注所有其他token。与使用因果掩码的GPT模型相比，模型准确率略有提升。
11. **在损失和反向传播中忽略填充索引（第1行vs第18行）**：设置`--ignore_index 50256`在PyTorch的`cross_entropy`损失函数中排除`<|endoftext|>`填充token。在这种情况下，它没有任何影响，因为替换了输出层，使得token ID在二元分类示例中为0或1。然而，当在第7章中对指令微调模型时，这个设置是有用的。
12. **对所有token的嵌入求平均（第1行vs第19行）**：设置`--average_embeddings`将对所有token的嵌入求平均。如果不使用此选项（默认），则只考虑所选token位置（由`--trainable_token_pos`指定）的输出嵌入；例如，最后一个token的嵌入。启用`--average_embeddings`将把所有token的嵌入平均池化到由`--trainable_token_pos`选择的位置（默认为最后一个token）。正如看到的，这从95.00%提高到96.33%，而运行时间仅略有增加（从0.28分钟到0.32分钟），在实践中可能是值得考虑的。
