# 故障排除指南

本页收集了在学习本书过程中遇到的常见问题和设置提示。

&nbsp;
## Notebook 图像加载问题

章节 Notebook 使用托管在 `https://sebastianraschka.com/images/LLMs-from-scratch-images/...` 的 Markdown 图像链接。这使得仓库下载大小可控，但也意味着图像依赖于图像主机和您的网络连接。

如果 `.ipynb` Notebook 中的图像无法渲染：

- 直接在浏览器中打开其中一个图像 URL，例如 [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp)。
- 如果 URL 在浏览器中也无法加载，问题可能是临时的网站、DNS、VPN、代理、防火墙或本地网络问题，而不是 Notebook 本身的问题。
- 我建议在不同设备或网络上再次检查 URL（尝试在手机上打开图像）；如果图像在手机上加载正常，则可能是您计算机上的 VPN 或防火墙问题。
- 如果您的手机上也无法加载图像，请随时在 GitHub 上提交 [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) 来帮助我进一步调试。

&nbsp;
## 在更新仓库的同时保留个人 Notebook 修改

如果您想在接收仓库更新的同时修改 Notebook，请先 Fork 仓库，然后克隆您的 Fork。主要配套 Notebook 与印刷书籍保持同步，通常不会更改，除非是关键修复。大多数仓库更新是添加额外材料。

Notebook 文件是 JSON 文件，因此 Git 差异和合并冲突可能难以阅读。为避免不必要的冲突，我建议将您的实验与受跟踪的配套 Notebook 分开：

- 在修改前复制一个 Notebook，例如从 `ch02.ipynb` 复制到 `ch02_experiments.ipynb`。
- 将您的草稿 Notebook 保存在单独的文件夹或您自己的分支上。
- 使用 `upstream` 远程仓库从原始仓库获取更新，仅在需要时合并或变基。

要创建 Fork 并克隆它：

1. 打开 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
2. 点击 GitHub 右上角的 **Fork** 按钮。
3. 克隆您的 Fork，将 `YOUR-USERNAME` 替换为您的 GitHub 用户名：

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

然后将原始仓库添加为 `upstream` 以便获取未来更新：

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

如果您确实需要合并编辑过的 Notebook，考虑安装 [`nbdime`](https://nbdime.readthedocs.io/) 以获得感知 Notebook 的差异和合并工具：

```bash
pip install nbdime
nbdime config-git --enable
```

有关更多背景信息，请参阅 [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)。

&nbsp;
## Apple Silicon 和 MPS 支持

某些 Notebook 和脚本在可用时使用 `cuda`，否则回退到 `cpu`，而不选择 Apple 的 `mps` 后端。许多地方故意省略 `mps` 支持，因为早期的 PyTorch/MPS 版本在多个示例中产生了不稳定或不同的结果，尤其是在训练和微调期间。

如果您使用的是 Apple Silicon Mac 并看到损失发散、损失尖峰、生成的文本质量差或与书籍不符的结果，请先在 `cpu` 上重新运行示例。为了获得与书籍匹配的更快训练行为，我建议在本地的 NVIDIA GPU 或云 GPU 上使用 `cuda`。

较新的 PyTorch 版本可能会改善 MPS 行为，如果您仔细验证结果，可以在本地尝试 `mps`。但是，如果您自己向脚本添加 `mps` 支持，请记住 CUDA 特定选项（如 `pin_memory=True`、`torch.compile` 和 DDP/多 GPU 代码）可能需要单独的保护措施。

有关更多背景信息，请参阅 [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977)、[#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625)、[#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644)、[#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) 和 [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846)。

&nbsp;
## 其他问题

对于其他问题，请随时在 GitHub 上提交新的 [Issue](https://github.com/rasbt/LLMs-from-scratch/issues)。
