# 预训练模型加载的替代方案

此路径包含备用的权重加载策略，以防权重从OpenAI下载的权重不可用。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb): （推荐）包含从由原始TensorFlow权重转换创建的PyTorch状态字典中加载权重的代码

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb): 包含通过`transformers`库从Hugging Face模型库加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb): 包含通过`safetensors`库直接从Hugging Face模型库加载权重的代码（跳过Hugging Face transformer模型的实例化）