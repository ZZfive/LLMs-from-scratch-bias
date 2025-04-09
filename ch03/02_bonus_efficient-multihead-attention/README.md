# 更多高效的多头注意力实现

- [mha-implementations.ipynb](mha-implementations.ipynb)包含和比较了不同多头注意力实现

### 总结

下图总结了性能基准测试结果，数值越低表示性能越好


&nbsp;
#### 仅前向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/1_forward-only.webp?1" width="500px"></a>

&nbsp;
#### 前向传播和反向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/2_forward-and-backward.webp?1" width="500px"></a>

&nbsp;
#### 前向传播和反向传播后编译

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/3_forward-and-backward-compiled.webp?1" width="500px"></a>