# CIFAR-100 Image Classification Experiments

PyTorch notebooks exploring **ResNet and Vision Transformer models on CIFAR-100**, with training curves, checkpoint saving, and image-perturbation experiments. The repository name is preserved as `cider-100`; the dataset used by the code is CIFAR-100.

These are archived experiment notebooks. They retain saved outputs, and the original ResNet notebook requires repairs before a clean, top-to-bottom execution.

## Notebook guide

| Notebook | Experiment | Configuration recorded in source |
| --- | --- | --- |
| [ResNet_cifar100.ipynb](ResNet_cifar100.ipynb) | ImageNet-pretrained ResNet-50 with a replacement classifier; prediction visualizations and perturbation evaluation | Batch size 64, 20 epochs, SGD, learning rate 0.003, StepLR |
| [ResNet_hyperparameterSearch.ipynb](ResNet_hyperparameterSearch.ipynb) | Configurable ResNet training with checkpointing and optional CUDA mixed precision | ResNet-101 by default, no pretrained weights, batch size 128, 100 epochs, SGD, learning rate 0.01, cosine schedule |
| [ViT_cifar100.ipynb](ViT_cifar100.ipynb) | ImageNet-pretrained ViT-B/16 fine-tuning with a 100-class head | Images resized to 224 pixels, batch size 128, 20 epochs, AdamW, learning rate 0.0001, cosine schedule |

Despite its filename, `ResNet_hyperparameterSearch.ipynb` configures one selected architecture per execution; it does not implement an automated search over a parameter grid.

## Open the notebooks

Use Python 3 and Jupyter. Imports reference PyTorch, torchvision, NumPy, Matplotlib, Pillow, and tqdm. There is no pinned environment file, so the following is a starting environment rather than an exact reconstruction of the original setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab torch torchvision numpy matplotlib pillow tqdm
jupyter lab
```

The notebooks download CIFAR-100 into `data/`. Models configured with pretrained weights also download those weights. CUDA is used when available in the ViT and configurable ResNet notebooks; CPU training can be substantially slower. Review epoch count, batch size, and worker count before launching a run.

See [reproducibility notes](docs/REPRODUCIBILITY.md) before interpreting results or running the original ResNet-50 notebook.

## Recorded outputs and their limits

The ViT notebook contains a saved final accuracy of **89.15%**. This is an output preserved in the notebook, not a result rerun or independently verified during documentation maintenance. The notebook chooses its best checkpoint using test-set accuracy, so that test set also acts as a model-selection set.

The original ResNet-50 notebook contains a saved **98.59%** accuracy with an evaluation count of **50,000 images**, while the dataset cell shows a separate test split of 10,000 images. Its dataloader uses an undefined `data_sets` variable instead of the defined `cifar100_datasets`. The saved value cannot be presented as a verified CIFAR-100 test benchmark.

The notebooks also differ in architecture, initialization, transforms, and training settings. Their saved outputs do not establish a controlled head-to-head comparison.

## Learning focus

- Adapting pretrained image classifiers to a 100-class task.
- Comparing training settings and learning-rate schedules.
- Recording accuracy/loss curves and model checkpoints.
- Exploring sensitivity to Gaussian noise, salt-and-pepper noise, brightness, contrast, and masked image patches.

## Archive contents

The repository contains three notebooks, including saved text and figure outputs. Dataset files and model checkpoints are not included. No repository-wide license or attribution record for a specific course/team is present; the documentation does not infer a course code or individual contribution breakdown.
