# CIFAR-100 Image Classification Experiments

**[Read the project report](Report.pdf)** · [Report details](docs/REPORT.md)

PyTorch experiments for adapting **ResNet and Vision Transformer classifiers to CIFAR-100**, recording training behaviour, and examining sensitivity to image perturbations. The repository includes three experiment workflows, preserved historical notebooks, and an offline execution check for the maintained ResNet-50 workflow.

**中文概述：** 本项目使用 ResNet 与 Vision Transformer 探索 CIFAR-100 图像分类，并分析噪声、亮度、对比度和局部遮挡对预测的影响。原始实验及输出完整保存在 `archive/`，整理后的笔记本放在 `notebooks/`。已修复 ResNet-50 的运行与统计问题；本次验证使用离线小样本，不代表新的分类准确率。

**Related paper:** [Deep Learning and Robotics](docs/REPORT.md), authored by Yongjiang Liu, was recovered from the Overleaf project *Cognitive Robotic*. Its original footnote links to `yongjiangliu-uom/cider-100`, identifying this project despite the different paper title. The preserved reading copy was compiled locally from the unchanged recovered source on 10 September 2026; its relationship to the notebooks and reported results is documented below.

## Project overview

The work covers dataset preparation, classifier adaptation, training and evaluation loops, checkpoint saving, learning curves, and robustness evaluation. It is useful for studying how these pieces fit together and for continuing individual experiments with explicit settings.

| Working notebook | Main workflow | Default configuration in the maintained code |
| --- | --- | --- |
| [ResNet-50](notebooks/ResNet_cifar100.ipynb) | ImageNet initialization, custom classifier, prediction plots, and five perturbation families | 32×32 inputs; batch 64; 20 epochs; SGD at 0.003; StepLR |
| [Configurable ResNet](notebooks/ResNet_hyperparameterSearch.ipynb) | Select a ResNet variant, train, plot curves, and save best/periodic checkpoints | ResNet-101 without pretraining; 224×224 inputs; batch 128; 100 epochs; SGD at 0.01; cosine schedule |
| [Vision Transformer](notebooks/ViT_cifar100.ipynb) | Fine-tune an ImageNet-pretrained ViT-B/16 with a 100-class head | 224×224 inputs; batch 128; 20 epochs; AdamW at 0.0001; cosine schedule |

The configurable ResNet notebook runs one selected configuration at a time. Its historical filename does not imply an automated hyperparameter search. The three configurations also differ in preprocessing and initialization, so their results are not a controlled architecture comparison.

## Repository structure

```text
notebooks/                   Maintained copies with cleared execution outputs
archive/original-notebooks/   Original notebooks, including historical outputs
archive/original_notebooks.json  Source revision and SHA-256 checksums
tests/test_notebooks.py       Offline integrity and ResNet-50 execution checks
docs/                        Reproduction, maintenance, and result interpretation
Report.pdf                   Final paper
latex/                       Paper source and figures
docs/                        Report provenance and limitations
requirements.txt             Working environment dependencies
requirements-test.txt        Additional notebook-validation dependency
```

Start with a working notebook to run an experiment, or open an [original notebook](archive/original-notebooks) to inspect historical evidence. Dataset downloads and generated checkpoints are excluded from version control.

## Getting started

Use Python 3.12 and an environment suitable for your CPU or CUDA installation. The recorded validation environment is listed in the [reproduction guide](docs/REPRODUCIBILITY.md#validated-environment).

```bash
git clone https://github.com/yongjiang-niuniu/cifar-100.git
cd cifar-100
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab notebooks/
```

Select the environment's Python kernel and run cells from the beginning. The notebooks download CIFAR-100 into a relative `data/` directory. Pretrained configurations also download model weights. Generated `data/` and `saved_models/` paths are relative to the notebook kernel's working directory.

Review batch size, epoch count, and worker count before a full run. CPU execution is supported by the ResNet-50 check, while the default 224-pixel experiments can require substantial memory and training time. The source chooses CUDA when available; Apple MPS is not selected automatically.

## Maintenance and validation

The ResNet-50 working copy now defines its transforms and device, uses consistent dataset/model names, computes sample-weighted mean loss, and binds each perturbation's severity correctly. Its sample visualization uses the same salt-and-pepper function as evaluation. The original files remain byte-for-byte intact; the new 32-pixel transforms are explicitly documented maintenance choices.

Run the offline checks from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Two checks passed during this refresh: notebook/archive integrity, and a one-epoch CPU execution of the actual ResNet-50 training and evaluation cells using 16 generated training images and 11 generated test images. The check verifies checkpoint creation, 100-class outputs, loss averaging with an uneven final batch, and all 25 perturbation conditions. Network access is blocked in the test; pretrained weights are disabled. This exercises program behaviour rather than measuring CIFAR-100 accuracy.

## Results and interpretation

| Evidence | What it supports |
| --- | --- |
| Original ViT output: **89.15%** accuracy | A preserved historical result. The notebook selects its checkpoint using the test set; the result was not reproduced during this refresh. |
| Original ResNet-50 output: **98.59%**, with **50,000** evaluated images | An inconsistent historical output: the declared test split contains 10,000 images and the old loader references an undefined dataset variable. It cannot establish test accuracy. |
| Offline maintenance checks | The documented ResNet-50 code paths execute on generated fixtures, and originals remain intact. No new benchmark result is claimed. |

ResNet-50 evaluates five levels of Gaussian noise, salt-and-pepper noise, brightness, contrast, and patch masking. These are 25 **conditions**: brightness and contrast at level 1 are identity settings. The final evaluation uses the in-memory model from the last epoch; it does not reload the checkpoint selected by minimum test loss.

The notebooks retain their test-set model-selection behaviour. A new performance study should introduce a separate validation split and reserve the test set for final evaluation. See [full limitations and repair details](docs/REPRODUCIBILITY.md).

The recovered paper describes ImageNet-21k pretraining, W&B sweeps and a ViT result of 87.23%. The available ViT notebook uses ImageNet-1k weights and preserves a different historical output of 89.15%; the configurable ResNet notebook does not contain an automated sweep. The paper includes W&B plots, but the complete sweep configuration and run exports have not been recovered. These are related project artifacts, not a verified one-to-one reconstruction of every reported experiment. See the [paper and notebook comparison](docs/REPORT.md#relationship-to-the-notebooks).

## Contribution and preservation

This repository is part of Yongjiang Liu's project collection. The recovered paper explicitly names Yongjiang Liu and links to the project's former repository address. Its Overleaf project title supplies the cognitive-robotics coursework context; the paper does not print a module code or establish a formal submission date. Original experiments and later maintenance remain distinguished in the preservation records.

The repository was renamed from `cider-100` to **`cifar-100`** to match the dataset. It is the same GitHub repository with its existing history retained. The [archive manifest](archive/original_notebooks.json) records the original source commit and checksums. No new license is granted for the notebooks, downloaded datasets, model weights, or dependencies by this documentation.

[Documentation guide](docs/README.md) · [Project portfolio](https://github.com/yongjiang-niuniu/academic-project-portfolio)
