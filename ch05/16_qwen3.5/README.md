# 从零构建 Qwen3.5 0.8B

这个文件夹包含 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的从零实现风格代码。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5 基于 Qwen3-Next 架构，我在 [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) 一文的 [2.（线性）注意力混合体](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) 小节中做了更详细的介绍。

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

需要注意的是，Qwen3.5 交替使用 `linear_attention` 和 `full_attention` 层。  
这些 notebook 在保持完整模型流程可读性的同时，复用了 [qwen3_5_transformers.py](qwen3_5_transformers.py) 中的线性注意力构件；该文件包含来自 Hugging Face 的线性注意力代码，并遵循 Apache 2.0 开源许可证。

&nbsp;
## 文件

- [qwen3.5.ipynb](qwen3.5.ipynb)：Qwen3.5 0.8B 的主 notebook 实现。
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb)：加入 KV-cache 解码的同模型版本，用于提升效率。
- [qwen3_5_transformers.py](qwen3_5_transformers.py)：Qwen3.5 线性注意力用到的部分 Hugging Face Transformers 辅助组件。
