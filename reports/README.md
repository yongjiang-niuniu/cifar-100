# Deep Learning and Robotics

**[Read the five-page paper](overleaf/2026-09-10/Cognitive_Robotic.pdf)** — Yongjiang Liu.

The paper combines a review of robotic perception, motion and interaction with CIFAR-100 experiments using ResNet and ViT-B/16. Its discussions of LLMs, reinforcement learning and embodied agents are literature context, not evidence that this project implemented or deployed those systems.

**中文概述：** 这份署名刘勇江的论文来自 Overleaf 项目 “Cognitive Robotic”，正文脚注直接指向本仓库原地址。报告讨论机器人感知背景及 CIFAR-100 分类实验；这里保留从原始源码本地编译的阅读版，不修改正文、署名或旧链接，也不将它视为所有笔记本实验的精确复现。

## Identity and status

| Item | Evidence |
| --- | --- |
| Printed title / author | *Deep Learning and Robotics* / Yongjiang Liu |
| Coursework context | Overleaf project named *Cognitive Robotic*; no module code printed in the paper |
| Project source | [Original Overleaf project](https://www.overleaf.com/project/67c8925df16050a563d041f9) |
| Historical code link | PDF page 3 links to `https://github.com/yongjiangliu-uom/cider-100.git` |
| Current repository | [yongjiang-niuniu/cifar-100](https://github.com/yongjiang-niuniu/cifar-100), formerly `cider-100` |
| Submission status | Authored coursework report; no official submission receipt was recovered with this export |
| PDF version | Local Tectonic build on 10 September 2026, not a downloaded Overleaf PDF |

The source ZIP includes an authored `PaperForReview.tex` manuscript and four experiment figures, alongside bibliography and template support files. A bundled CVPR-template README and rebuttal template describe those support files; they do not change the manuscript's explicit author identity.

## Relationship to the notebooks

| Topic | Recovered paper | Available notebook evidence |
| --- | --- | --- |
| Dataset and architectures | CIFAR-100, ResNet family and ViT-B/16 | Same dataset and model families, supporting the explicit repository link |
| Pretraining | Describes ImageNet-21k pretraining | The ViT notebook selects Torchvision ImageNet-1k weights; its configuration does not substantiate the 21k statement |
| Parameter exploration | Describes W&B sweeps and includes parallel-coordinate plots | The configurable ResNet notebook runs one chosen configuration; complete sweep definitions and run exports are absent |
| ViT result | Table IV reports 87.23% at 224 × 224 | The original ViT notebook preserves 89.15%; the different figures have not been reconciled to matching checkpoints/runs |
| ResNet results | Table III reports several variants, including pretrained ResNet-50 at 84.60% | The original ResNet-50 output reports 98.59% over 50,000 evaluated images, inconsistent with its declared test split; it does not verify the paper's result |
| Architecture comparison | Includes official literature results at other resolutions and discusses a fairer comparison as future work | Notebook configurations differ in resolution, initialization and training settings; neither artifact establishes a controlled comparison under identical conditions |

Reported values are preserved historical statements. No training or benchmark evaluation was run while recovering this paper. Reproducing individual tables would require the original run configurations, data splits, weights and logs. The documentation does not choose whichever historical number is most favourable.

## Build and preservation

Overleaf compilation timed out during recovery, so the downloaded source was checked for unsafe archive paths and extracted into an isolated audit directory. All ten original files were preserved. A separate build copy compiled successfully with Tectonic without changing any source file or body text. The original source ZIP remains in the local recovery archive; it is not included in this repository update.

| Artifact | SHA-256 / size |
| --- | --- |
| Original `Cognitive_Robotic.zip` | `3eb787e163464b50c932c4090c5c8b38ed5ceba09d4312fa0680f53e29644f0a` / 6,189,212 bytes |
| Local reading PDF | `cbda6b701d536b633e27c0a23754be2479507eedb9df271a6f5777030b5bc80f` / 5,584,797 bytes / 5 pages |

All five PDF pages were visually checked. Tables, figures and text are readable without page clipping or overlap. Existing source issues remain visible: the missing `mehta2021mobilevit` bibliography entry produces `[?]` on page 4; one bibliography entry has an empty journal field, and the compiler reports paragraph-spacing and a small table-width warning. These were recorded rather than silently altering the original report. The old repository URL and author name are unchanged.

[Back to the project](../README.md) · [Notebook reproduction notes](../docs/REPRODUCIBILITY.md)
