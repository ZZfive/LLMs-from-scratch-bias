# 构建与预训练LLM交互的用户界面

这个附加文件夹包含运行类似ChatGPT的用户界面的代码，用于与第5章中预训练的LLMs进行交互，如下所示。

![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)

要实现这个用户界面，使用开源的[Chainlit](https://github.com/Chainlit/chainlit)Python包。

&nbsp;
## 步骤1: 安装依赖

首先，安装使用以下命令安装`chainlit`

```bash
pip install chainlit
```

（或者，执行`pip install -r requirements-extra.txt`）

&nbsp;
## 步骤2: 安装依赖

这个文件夹包含2个文件：
1. [`app_orig.py`](app_orig.py)：该文件加载并使用OpenAI的原始GPT-2权重。
2. [`app_own.py`](app_own.py)：该文件加载并使用在第五章生成的GPT-2权重。这要求你首先执行[`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb)文件。

(打开并检查这些文件以了解更多。)

从终端运行以下命令之一以启动UI服务器：

```bash
chainlit run app_orig.py
```

或

```bash
chainlit run app_own.py
```

运行上述命令之一应该会打开一个新浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请检查终端命令并将本地地址复制到你的浏览器地址栏（通常，地址是`http://localhost:8000`）。