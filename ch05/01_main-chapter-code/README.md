# Chapter 5: Pretraining on Unlabeled Data

### Main Chapter Code

- [ch05.ipynb](ch05.ipynb) 包含本章节主要代码
- [previous_chapters.py](previous_chapters.py) 是一个Python模块，包含第3章中的`MultiHeadAttention`模块和`GPTModel`类，在[ch05.ipynb](ch05.ipynb)中导入它来预训练GPT模型
- [gpt_download.py](gpt_download.py) 包含下载预训练GPT模型权重的实用函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章节的练习解决方案

### Optional Code

- [gpt_train.py](gpt_train.py) 是一个独立的Python脚本文件，包含在[ch05.ipynb](ch05.ipynb)中实现的代码，用于训练GPT模型（你可以把它看作是本章节的代码总结）
- [gpt_generate.py](gpt_generate.py) 是一个独立的Python脚本文件，包含在[ch05.ipynb](ch05.ipynb)中实现的代码，用于加载和使用OpenAI的预训练模型权重

