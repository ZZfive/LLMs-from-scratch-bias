# Chapter 5: Pretraining on Unlabeled Data

&nbsp;
## Main Chapter Code

- [01_main-chapter-code](01_main-chapter-code) 包含本章节主要代码

&nbsp;
## Bonus Materials

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含从备选位置加载GPT模型权重的代码，以防模型权重无法从OpenAI获取
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含用于在Project Gutenberg提供的整个书籍语料库上对LLM进行更长时间的预训练的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含实现更复杂训练功能的代码，包括学习率调度器和梯度裁剪
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含一个可选的超参数调整脚本
- [06_user_interface](06_user_interface) 实现了一个交互式用户界面，用于与预训练的LLM交互
- [07_gpt_to_llama](07_gpt_to_llama) 包含将GPT架构实现转换为Llama 3.2的逐步指南，并从Meta AI加载预训练权重
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个附加笔记本，展示了如何通过PyTorch的load_state_dict方法更高效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含了从零开始的GPT-2 BPE分词器的实现
- [10_llm-training-speed](10_llm-training-speed) 展示了PyTorch性能技巧以提升LLM训练速度
- [11_qwen3](11_qwen3) 一个从零开始的Qwen3 0.6B和Qwen3 30B-A3B（专家混合）实现，包括加载基础、推理和编码模型变体预训练权重的代码
- [12_gemma3](12_gemma3) 一个从头开始实现的Gemma 3 270M及其替代方案，包括使用KV缓存的代码，以及加载预训练权重的代码

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)