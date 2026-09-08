# Reproducibility and result interpretation

This document records limitations visible in the committed notebooks. No training run was launched as part of documentation maintenance, and historical outputs have been retained.

## Original ResNet-50 notebook

`ResNet_cifar100.ipynb` depends on notebook state that is not fully represented in the saved cells:

| Issue | Evidence and consequence |
| --- | --- |
| Missing transform definition | `data_preprocessing_transforms` is used when creating datasets but is never defined in the notebook. |
| Inconsistent dataset names | Dataset cells assign `cifar100_datasets`; dataloaders and later visualizations use `data_sets`, which is not defined. |
| Missing device flag | `use_cuda` is used without an assignment. |
| Missing model variable | Perturbation evaluation passes `model`, whereas the classifier is assigned to `resnet50_model`. |
| Debug cell depends on old state | `phase`, `correct_predictions`, and `dataset_size` are printed before the training loop; `dataset_size` is not assigned in the saved code. |
| Repeated dataset cell | Two consecutive cells create the same `cifar100_datasets` mapping. |
| Inconsistent evaluation count | A saved evaluation reports 50,000 test images, while the dataset description records 10,000 in its test split. |

Consequently, restarting the kernel and running all cells will fail before training. Missing definitions should be restored and dataset names made consistent before a new evaluation is attempted. The archived outputs alone cannot determine which data the historical kernel actually used.

The training loop accumulates batch-mean cross-entropy losses and divides by the number of samples. This is not the usual sample-weighted mean loss; its numerical scale should not be compared directly with correctly averaged loss values.

The perturbation-loader factory also creates lambdas inside a severity loop without binding each loop value. Gaussian, salt-and-pepper, and mask transforms may therefore use the final loop value during later evaluation. Per-severity robustness claims require correcting and rerunning this code.

`np.Inf` and `models.resnet50(pretrained=True)` are historical API usages. With an unpinned environment, compatibility must be checked before execution.

## Configurable ResNet notebook

`ResNet_hyperparameterSearch.ipynb` offers ResNet-18, -34, -50, and -101 through `get_model`. Its committed settings select ResNet-101 with `pretrained=False` and a 100-epoch run.

It saves best and periodic checkpoints, records optimizer/scheduler state, and includes loss/accuracy plots. CUDA cache handling and mixed precision are present. The source seeds PyTorch and NumPy with 42, but this does not establish bit-for-bit reproducibility across devices and software versions.

The name refers to manually configurable experiments; there is no automated grid or Bayesian search. Changing parameters and rerunning is a separate experiment that should be recorded explicitly.

## Vision Transformer notebook

`ViT_cifar100.ipynb` uses torchvision's ViT-B/16 with ImageNet weights, replaces its head for 100 classes, and resizes CIFAR images to 224 pixels. It uses the official train/test dataset flags and records a saved best-model accuracy of 89.15%.

The test set is evaluated every epoch and selects the saved checkpoint. A future benchmark should select models using a validation subset of the training data, then evaluate the test set after model selection. The current result is an archived selected-checkpoint output with that limitation.

## Before recording a new result

1. Restart the kernel and verify every variable is defined by the notebook itself.
2. Confirm the training, validation, and test splits and their sample counts before constructing loaders.
3. Record Python, PyTorch, torchvision, CUDA, hardware, seed, transforms, and all hyperparameters.
4. Keep model selection separate from final test evaluation, and distinguish pretrained from randomly initialized models.
5. Save the exact executed notebook or code revision, split definition, and machine-readable metrics alongside a run identifier.

The existing notebooks are useful records of experimentation, but their saved metrics should retain these qualifications in portfolios, reports, and future comparisons.
