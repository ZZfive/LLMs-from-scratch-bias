# 从零构建 Qwen3 与聊天界面



这个 bonus 文件夹包含用于运行类 ChatGPT 用户界面的代码，可用来与预训练的 Qwen3 模型交互。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



这个用户界面基于开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit) 实现。

&nbsp;
## 步骤1：安装依赖

首先，通过下面的命令安装 `chainlit` 包以及 [requirements-extra.txt](requirements-extra.txt) 中列出的依赖：

```bash
pip install -r requirements-extra.txt
```

或者，如果使用`uv`：

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## 步骤2：运行`app`代码

此文件夹包含 2 个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：加载并以思维模式使用 Qwen3 0.6B 模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上面类似，但配置为保留消息历史。

（你可以直接打开这些文件查看更详细的实现。）

在终端中运行下面任一命令来启动 UI 服务器：

```bash
chainlit run qwen3-chat-interface.py
```

或者，如果使用`uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述任一命令后，通常会自动打开一个新的浏览器标签页，你可以在其中与模型交互。如果浏览器没有自动打开，请查看终端输出，并将本地地址复制到浏览器地址栏中（通常是 `http://localhost:8000`）。
