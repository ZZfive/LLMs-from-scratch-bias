# 从零构建 Tiny Aya 3.35B

Tiny Aya 是 Cohere 推出的一款新的“小型”LLM，据称它是 30 亿参数级别中“能力最强的多语言开放权重模型”。根据其[发布公告](https://cohere.com/blog/cohere-labs-tiny-aya)，Tiny Aya 的表现超过了 Qwen3-4B、Gemma 3 4B 和 Ministral 3 3B。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">

这是一个很适合在本地运行和实验的模型。唯一需要注意的是，虽然它是开放权重模型，但许可证限制相对较多，只允许非商业用途。

除此之外，Arya 是一个 3.35B 参数模型，提供多个适合个人和（非商业）研究用途的版本：

- [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base)（基础模型）
- [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)（跨语言与地区的综合平衡最好；本 notebook 默认使用）
- [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)（针对南亚语言优化）
- [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)（针对欧洲和亚太语言优化）
- [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth)（针对西亚和非洲语言优化）

更具体地说，下表列出了这些模型重点优化的语言：

| 地区 | 语言 | 优化模型 |
| --- | --- | --- |
| **亚太** | 繁体中文、粤语、越南语、他加禄语、爪哇语、高棉语、泰语、缅甸语、马来语、韩语、老挝语、印尼语、简体中文、日语 | tiny-aya-water |
| **非洲** | 祖鲁语、阿姆哈拉语、豪萨语、伊博语、斯瓦希里语、科萨语、沃洛夫语、绍纳语、约鲁巴语、尼日利亚皮钦语、马达加斯加语 | tiny-aya-earth |
| **南亚** | 泰卢固语、马拉地语、孟加拉语、泰米尔语、印地语、旁遮普语、古吉拉特语、乌尔都语、尼泊尔语 | tiny-aya-fire |
| **欧洲** | 加泰罗尼亚语、加利西亚语、荷兰语、丹麦语、芬兰语、捷克语、葡萄牙语、法语、立陶宛语、斯洛伐克语、巴斯克语、英语、瑞典语、波兰语、西班牙语、斯洛文尼亚语、乌克兰语、希腊语、挪威语 Bokmal、罗马尼亚语、塞尔维亚语、德语、意大利语、俄语、爱尔兰语、匈牙利语、保加利亚语、克罗地亚语、爱沙尼亚语、拉脱维亚语、威尔士语 | tiny-aya-water |
| **西亚** | 阿拉伯语、马耳他语、土耳其语、希伯来语、波斯语 | tiny-aya-earth |

从架构角度看，Tiny Aya 是经典的 decoder-only transformer，但包含几个值得注意的改动（除了 SwiGLU 和 Grouped Query Attention 这类已经比较常见的设计）：

1. **并行 Transformer block。** 并行块会基于同一份归一化输入同时计算 attention 和 MLP，然后一步加回残差。我推测这是为了减少层内串行依赖，从而提升计算吞吐。
2. **滑动窗口注意力。** 具体来说，它采用了类似 Arcee Trinity 和 Olmo 3 的 3:1 局部:全局比例，窗口大小也是 4096。并且和 Arcee 类似，滑动窗口层使用 RoPE，而全注意力层使用 NoPE。
3. **LayerNorm。** 许多架构已经转向 RMSNorm，因为它计算更便宜且效果不错。Tiny Aya 则保留了更经典的 LayerNorm 风格，不过这里用的是一个改写版本，没有 shift，也就是没有 bias 参数。

&nbsp;
## 文件

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) 是一个独立的 Jupyter notebook，完整实现了 Tiny Aya 架构并加载预训练权重。

另一个 [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) notebook 在此基础上加入了 KV cache，以提升运行时性能（但也会增加代码复杂度）。如果你想进一步了解 KV cache，可参考我的文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

<br>

如果你想进一步了解不同架构之间的差异，以及与其他架构的对比，可以阅读我的文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。
