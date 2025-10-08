# 在Project Gutenberg数据集上预训练 GPT

本目录中的代码包含使用Project Gutenberg提供的免费书籍训练小型GPT模型的代码。

正如Project Gutenberg网站所说，“绝大多数Project Gutenberg电子书在美国属于公共领域。”

请查阅[Project Gutenberg 的权限、许可及其他常见请求页面](https://www.gutenberg.org/policy/permission.html)，以获取更多关于使用 Project Gutenberg 提供的资源的信息。

&nbsp;
## 如何使用这个代码

&nbsp;
### 1) 下载数据集

在本节中，使用[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg)仓库中的代码从古腾堡计划下载书籍。

截至本次写作，这将需要大约50 GB的磁盘空间，并花费大约10-15小时，但具体时间可能更多，取决于Project Gutenberg自那时以来的增长情况。

&nbsp;

#### Linux和macOS用户的下载说明


Linux和macOS用户可以按照以下步骤下载数据集（如果是Windows用户，请查看下面的说明）：

1. 将`03_bonus_pretraining_on_gutenberg`文件夹设置为工作目录，以便在此文件夹中本地克隆`gutenberg`仓库（这是运行提供的脚本`prepare_dataset.py`和`pretraining_simple.py`的必要条件）。例如，当位于`LLMs-from-scratch`仓库的文件夹中时，可以通过以下方式进入*03_bonus_pretraining_on_gutenberg*文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在其中克隆`gutenberg`仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的`gutenberg`仓库的文件夹：
```bash
cd gutenberg
```

4. 从`gutenberg`仓库的文件夹中安装*requirements.txt*中定义的所需包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 回到`03_bonus_pretraining_on_gutenberg`文件夹
```bash
cd ..
```

&nbsp;
#### Windows 用户特殊说明

`pgcorpus/gutenberg`代码与Linux和macOS兼容。然而，Windows用户需要做一些小的调整，例如在`subprocess`调用中添加`shell=True`并替换`rsync`。

或者，在Windows上运行此代码的一个更简单的方法是使用"Windows system for Linux"（WSL）功能，该功能允许用户在Windows中使用Ubuntu运行Linux环境。欲了解更多信息，请阅读[微软的官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)和[教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

在使用WSL时，请确保已安装Python 3（可通过`python3 --version`检查，或例如使用`sudo apt-get install -y python3.10`安装Python 3.10），并在其中安装以下包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意：**
> 关于如何设置Python和安装包的说明，请参考[可选的Python设置偏好](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/README.md)和[安装Python库](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/02_installing-python-libraries/README.md)。
>
> 可选地，本仓库提供了一个运行Ubuntu的Docker镜像。有关如何使用提供的Docker镜像运行容器的说明，请参阅[Optional Docker Environment](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/03_optional-docker-environment/README.md)。

&nbsp;
### 2) 准备数据集

接下来，运行`prepare_dataset.py`脚本，该脚本将（截至本文写作时，共60,173个）文本文件合并成较少的几个大文件，以便更高效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 请注意，生成的文件以纯文本格式存储，为了简化操作，并未进行预分词。但是，如果你计划更频繁地使用该数据集或进行多个周期的训练，你可能需要更新代码以将数据集存储为预分词形式，以节省计算时间。有关更多信息，请参阅本页底部的“设计决策与改进”。

> **提示：**
> 你可以选择更小的文件大小，例如50MB。这将导致文件数量增加，但对于在少量文件上快速进行预训练测试可能很有用。

&nbsp;
### 3) 运行预训练脚本

可以按照以下方式运行预训练脚本。请注意，附加的命令行参数以默认值显示，仅用于说明目的：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将按照以下格式显示：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...

&nbsp;
> **提示：**
> 在实际使用中，如果你使用的是macOS或Linux，建议除了在终端打印日志输出外，还使用`tee`命令将其保存到`log.txt`文件中：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
请注意，在`gutenberg_preprocessed`文件夹中的~500Mb文本文件之一上进行训练，在V100 GPU上大约需要4小时。该文件夹包含47个文件，完成全部训练大约需要200小时（超过1周）。您可能想在小得多的文件数量上运行它。

&nbsp;
## 设计决策与改进

请注意，此代码侧重于保持简单和精简，以用于教育目的。为了提高建模性能和训练效率，代码可以在以下方面进行改进：

1. 修改`prepare_dataset.py`脚本，以从每个书籍文件中去除Gutenberg模板文本。

2. 更新数据准备和加载工具，以便对数据集进行预分词，并将其以分词形式保存，这样在调用预训练脚本时就不必每次都重新分词。

3. 更新`train_model_simple`脚本，添加[附录D中介绍的特性](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)：余弦衰减、线性预热和梯度裁剪。

4. 更新预训练脚本以保存优化器状态（参见第5章第5.4节“在PyTorch中加载和保存权重”；ch05.ipynb），并添加选项以加载现有的模型和优化器检查点，如果训练运行中断，则继续训练。

5. 添加一个更高级的日志记录器（例如，Weights and Biases），以便实时查看损失和验证曲线

6. 添加分布式数据并行性（DDP），并在多个GPU上训练模型（参见附录A中的 A.9.3 节“使用多个GPU进行训练”；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。

7. 将`previous_chapter.py`脚本中从头实现的多头注意力类（MultiheadAttention）替换为[高效多头注意力实现附加章节](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)中通过`PyTorch`的`nn.functional.scaled_dot_product_attention`函数集成`Flash Attention`的`MHAPyTorchScaledDotProduct`类。

8. 通过[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或[thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型来加快训练速度

9. 实现梯度低秩投影（GaLore）以进一步加快预训练过程。这可以通过仅将`AdamW`优化器替换为[GaLore Python](https://github.com/jiaweizzhao/GaLore)库中提供的`GaLoreAdamW`来实现。