# 第7章：用于遵循指令的微调

此文件夹包含可用于模型评估的实用代码。



&nbsp;
## 使用OpenAI API评估指令响应


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb)笔记本使用OpenAI的GPT-4来评估指令微调模型生成的响应。它使用以下格式的JSON文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- 测试集中给定的目标
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- LLM的响应
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- 第二个LLM的响应
},
```

&nbsp;
## 使用Ollama本地评估指令响应

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb)笔记本提供了上述的替代方案，通过Ollama使用本地下载的Llama 3模型。