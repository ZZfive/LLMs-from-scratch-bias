# 从零构建Qwen3与聊天界面



此奖励文件夹包含用于运行类似ChatGPT的用户界面以与预训练的Qwen3模型交互的代码。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



为了实现此用户界面，我们使用开源的[Chainlit Python包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤1：安装依赖

首先，我们通过以下方式安装`chainlit`包和[requirements-extra.txt](requirements-extra.txt)列表中的依赖

```bash
pip install -r requirements-extra.txt
```

或者，如果您正在使用`uv`：

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## 步骤2：运行`app`代码

此文件夹包含2个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：此文件加载并以思维模式使用Qwen3 0.6B模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上面相同，但配置为记住消息历史。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一来启动UI服务器：

```bash
chainlit run qwen3-chat-interface.py
```

或者，如果您正在使用`uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述命令之一应该会打开一个新的浏览器选项卡，您可以在其中与模型交互。如果浏览器选项卡没有自动打开，请检查终端命令并将本地地址复制到浏览器地址栏中（通常，地址是`http://localhost:8000`）。
