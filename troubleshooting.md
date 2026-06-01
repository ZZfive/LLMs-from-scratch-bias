# 故障排除指南

本页收集了在学习本书过程中遇到的常见问题和设置提示。

## 笔记本图像加载问题

章节笔记本使用托管在 `https://sebastianraschka.com/images/LLMs-from-scratch-images/...` 的 Markdown 图像链接。这使仓库下载大小保持可控，但也意味着图像依赖于图像主机和网络连接。

如果 `.ipynb` 笔记本中的图像未渲染：

- 直接在浏览器中打开其中一个图像 URL，例如 [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp)。
- 如果浏览器也无法加载该 URL，问题很可能是临时网站、DNS、VPN、代理、防火墙或本地网络问题，而不是笔记本问题。
- 我建议在不同设备或网络上重新检查 URL（例如尝试在手机上打开图像）；如果图像在手机上能正常加载，则很可能指向计算机上的 VPN 或防火墙问题。
- 如果图像在手机上也无法加载，请随时打开 GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) 以帮助我进一步调试。

## 在更新仓库时保留个人笔记本更改

如果您希望在修改笔记本的同时接收仓库更新，请先分叉仓库，然后克隆您的分叉。主书笔记本与印刷版保持同步，通常不会更改，除非是紧急修复。大多数仓库更新都添加了补充材料。

笔记本文件是 JSON 文件，因此 Git 差异和合并冲突可能难以阅读。为避免不必要的冲突，我建议将实验与受跟踪的书笔记本分开：

- 在修改前复制笔记本，例如从 `ch02.ipynb` 到 `ch02_experiments.ipynb`。
- 将草稿笔记本放在单独的文件夹或自己的分支上。
- 从原始仓库使用 `upstream` 远程获取更新，仅在需要这些更新时进行合并或变基。

要创建分叉并克隆它：

1. 打开 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
2. 在 GitHub 右上角点击 **Fork** 按钮。
3. 克隆您的分叉，将 `YOUR-USERNAME` 替换为您的 GitHub 用户名：

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

然后将原始仓库添加为 `upstream`，以便获取未来更新：

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

如果您确实需要合并编辑过的笔记本，考虑安装 [`nbdime`](https://nbdime.readthedocs.io/) 以获得笔记本感知的差异和合并工具：

```bash
pip install nbdime
nbdime config-git --enable
```

更多上下文，请参阅 [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)。

## Apple Silicon 和 MPS 支持

一些笔记本和脚本在可用时使用 `cuda`，否则回退到 `cpu`，而没有选择 Apple 的 `mps` 后端。在许多地方，这种对 `mps` 支持的缺失是故意的，因为早期的 PyTorch/MPS 版本在多个示例中产生了不稳定或不同的结果，尤其是在训练和微调期间。

如果您使用 Apple Silicon Mac 并看到损失发散、损失尖峰、生成的文本质量差或与书不匹配的结果，请首先在 `cpu` 上重新运行示例。为了获得与书匹配行为的更快训练，我建议使用本地 NVIDIA GPU 或云 GPU 上的 `cuda`。

较新的 PyTorch 版本可能会改善 MPS 行为，如果您仔细验证结果，可以在本地尝试 `mps`。但是，如果您自己为脚本添加 `mps` 支持，请注意 CUDA 特定选项（如 `pin_memory=True`、`torch.compile` 和 DDP/多 GPU 代码）可能需要单独的保护。

更多上下文，请参阅 [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977)、[#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625)、[#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644)、[#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) 和 [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846)。

## 其他问题

对于其他问题，请随时打开新的 GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues)。
