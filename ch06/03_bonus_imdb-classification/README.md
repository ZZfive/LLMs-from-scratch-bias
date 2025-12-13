# 额外实验：对5万条IMDb电影评论进行情感分类

## 概述

此文件夹包含额外的实验，用于将第6章中的（解码器风格）GPT-2 (2018)模型与编码器风格的LLM如[BERT (2018)](https://arxiv.org/abs/1810.04805)、[RoBERTa (2019)](https://arxiv.org/abs/1907.11692)和[ModernBERT (2024)](https://arxiv.org/abs/2412.13663)进行比较。没有使用第6章中的小型垃圾邮件数据集，而是使用了来自IMDb的5万条电影评论数据集（[数据集来源](https://ai.stanford.edu/~amaas/data/sentiment/)），目标是二元分类，预测评论者是否喜欢这部电影。这是一个平衡的数据集，所以随机预测应该达到50%的准确率。





|       | Model                        | Test accuracy |
| ----- | ---------------------------- | ------------- |
| **1** | 124M GPT-2 Baseline          | 91.88%        |
| **2** | 340M BERT                    | 90.89%        |
| **3** | 66M DistilBERT               | 91.40%        |
| **4** | 355M RoBERTa                 | 92.95%        |
| **5** | 304M DeBERTa-v3              | 94.69%        |
| **6** | 149M ModernBERT Base         | 93.79%        |
| **7** | 395M ModernBERT Large        | 95.07%        |
| **8** | Logistic Regression Baseline | 88.85%        |






&nbsp;
## 步骤1：安装依赖

通过以下方式安装额外的依赖

```bash
pip install -r requirements-extra.txt
```

&nbsp;
## 步骤2：下载数据集

代码使用来自IMDb的5万条电影评论（[数据集来源](https://ai.stanford.edu/~amaas/data/sentiment/)）来预测电影评论是积极的还是消极的。

运行以下代码来创建`train.csv`、`validation.csv`和`test.csv`数据集：

```bash
python download_prepare_dataset.py
```


&nbsp;
## 步骤3：运行模型

&nbsp;
### 1) 124M GPT-2基线

第6章中使用的124M GPT-2模型，从预训练权重开始，微调所有权重：

```bash
python train_gpt.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 3.706, Val loss 3.853
Ep 1 (Step 000050): Train loss 0.682, Val loss 0.706
...
Ep 1 (Step 004300): Train loss 0.199, Val loss 0.285
Ep 1 (Step 004350): Train loss 0.188, Val loss 0.208
Training accuracy: 95.62% | Validation accuracy: 95.00%
Training completed in 9.48 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.64%
Validation accuracy: 92.32%
Test accuracy: 91.88%
```


<br>

---

<br>

&nbsp;
### 2) 340M BERT


一个3.4亿参数的编码器风格[BERT](https://arxiv.org/abs/1810.04805)模型：

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "bert"
```

```
Ep 1 (Step 000000): Train loss 0.848, Val loss 0.775
Ep 1 (Step 000050): Train loss 0.655, Val loss 0.682
...
Ep 1 (Step 004300): Train loss 0.146, Val loss 0.318
Ep 1 (Step 004350): Train loss 0.204, Val loss 0.217
Training accuracy: 92.50% | Validation accuracy: 88.75%
Training completed in 7.65 minutes.

Evaluating on the full datasets ...

Training accuracy: 94.35%
Validation accuracy: 90.74%
Test accuracy: 90.89%
```

<br>

---

<br>

&nbsp;
### 3) 66M DistilBERT

一个6.6亿参数的编码器风格[DistilBERT](https://arxiv.org/abs/1910.01108)模型（从3.4亿参数的BERT模型蒸馏而来），从预训练权重开始，只训练最后一个Transformer块和输出层：



```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "distilbert"
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.688
Ep 1 (Step 000050): Train loss 0.452, Val loss 0.460
...
Ep 1 (Step 004300): Train loss 0.179, Val loss 0.272
Ep 1 (Step 004350): Train loss 0.199, Val loss 0.182
Training accuracy: 95.62% | Validation accuracy: 91.25%
Training completed in 4.26 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.30%
Validation accuracy: 91.12%
Test accuracy: 91.40%
```
<br>

---

<br>

&nbsp;
### 4) 355M RoBERTa

一个3.55亿参数的编码器风格[RoBERTa](https://arxiv.org/abs/1907.11692)模型，从预训练权重开始，只训练最后一个Transformer块和输出层：


```bash
python train_bert_hf.py --trainable_layers "last_block" --num_epochs 1 --model "roberta"
```

```
Ep 1 (Step 000000): Train loss 0.695, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.670, Val loss 0.690
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.098
Ep 1 (Step 004350): Train loss 0.170, Val loss 0.086
Training accuracy: 98.12% | Validation accuracy: 96.88%
Training completed in 11.22 minutes.

Evaluating on the full datasets ...

Training accuracy: 96.23%
Validation accuracy: 94.52%
Test accuracy: 94.69%
```

<br>

---

<br>

&nbsp;
### 5) 304M DeBERTa-v3

一个3.04亿参数的编码器风格[DeBERTa-v3](https://arxiv.org/abs/2111.09543)模型。DeBERTa-v3通过解耦注意力机制和改进的位置编码在早期版本基础上有所改进。


```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "deberta-v3-base"
```

```
Ep 1 (Step 000000): Train loss 0.689, Val loss 0.694
Ep 1 (Step 000050): Train loss 0.673, Val loss 0.683
...
Ep 1 (Step 004300): Train loss 0.126, Val loss 0.149
Ep 1 (Step 004350): Train loss 0.211, Val loss 0.138
Training accuracy: 92.50% | Validation accuracy: 94.38%
Training completed in 7.20 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.44%
Validation accuracy: 93.02%
Test accuracy: 92.95%
```

<br>

---

<br>



&nbsp;
### 6) 149M ModernBERT Base

[ModernBERT (2024)](https://arxiv.org/abs/2412.13663) 是BERT的优化重新实现，包含了并行残差连接和门控线性单元（GLU）等架构改进，以提高效率和性能。它保持了BERT的原始预训练目标，同时在现代硬件上实现更快的推理和更好的可扩展性。

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-base"
```



```
Ep 1 (Step 000000): Train loss 0.699, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.564, Val loss 0.606
...
Ep 1 (Step 004300): Train loss 0.086, Val loss 0.168
Ep 1 (Step 004350): Train loss 0.160, Val loss 0.131
Training accuracy: 95.62% | Validation accuracy: 93.75%
Training completed in 10.27 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.72%
Validation accuracy: 94.00%
Test accuracy: 93.79%
```

<br>

---

<br>


&nbsp;
### 7) 395M ModernBERT Large

与上面相同，但使用更大的ModernBERT变体。

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-large"
```



```
Ep 1 (Step 000000): Train loss 0.666, Val loss 0.662
Ep 1 (Step 000050): Train loss 0.548, Val loss 0.556
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.115
Ep 1 (Step 004350): Train loss 0.154, Val loss 0.116
Training accuracy: 96.88% | Validation accuracy: 95.62%
Training completed in 27.69 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.04%
Validation accuracy: 95.30%
Test accuracy: 95.07%
```





<br>

---

<br>

&nbsp;
### 8) 逻辑回归基线

使用scikit-learn [逻辑回归](https://sebastianraschka.com/blog/2022/losses-learned-part1.html)分类器作为基线：


```bash
python train_sklearn_logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.62%
Test Accuracy: 88.85%
```
