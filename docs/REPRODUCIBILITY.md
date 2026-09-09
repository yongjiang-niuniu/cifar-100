# Reproduction and maintenance

The maintained notebooks separate runnable working copies from the original experiment evidence. They do not reconstruct an undocumented historical environment or establish new benchmark results.

**中文概述：** 原件及历史输出保持不变。本次为 ResNet-50 补齐缺失定义，修正损失统计与扰动级别，并使用生成的小样本完成离线运行检查。其他两本只整理格式和输出，没有重新训练。

## Original evidence

The three original notebooks were preserved byte-for-byte from commit `86b7d485cc56b79da22be990e5bb9ba35525c642` in the repository formerly named `cider-100`. Their paths and SHA-256 checksums are recorded in the [manifest](../archive/original_notebooks.json).

All working copies have cleared outputs and execution counts, a maintenance introduction, and original cell indices in metadata. The configurable ResNet and ViT copies retain their original code. Historical outputs remain available only in the archive, keeping them distinguishable from any future execution of the maintained code.

## ResNet-50 repairs

| Original issue | Maintained behaviour |
| --- | --- |
| `data_preprocessing_transforms` never defined; dataset construction duplicated | Replace the duplicate cell with explicit train/test transforms. |
| `data_sets` used instead of the defined `cifar100_datasets` | Use one dataset mapping for loaders, class counts, and plots. |
| Undefined `use_cuda`, debug state, and evaluation `model` | Resolve CPU/CUDA explicitly, show actual split lengths, and evaluate `resnet50_model`. |
| Shell-only directory creation and historical `np.Inf`/`pretrained=True` calls | Use `Path.mkdir`, `np.inf`, and the explicit torchvision weight enum. |
| Batch-mean losses summed then divided by sample count | Multiply each batch mean by its sample count before averaging. |
| Severity-loop lambdas read the final loop value | Capture salt-and-pepper/mask parameters at construction and instantiate Gaussian noise per level. |
| Image denormalization differs from evaluation normalization | Use the same CIFAR constants for the prediction display. |
| Salt-and-pepper illustration uses a different severity scale | Reuse the evaluation function with severity `0.05 × level`. |

The new train transform is random crop at 32 pixels with padding 4, random horizontal flip, tensor conversion, and normalization. The test transform is tensor conversion and normalization. Means are `(0.5071, 0.4865, 0.4409)` and standard deviations are `(0.2673, 0.2564, 0.2761)`. These are explicit maintenance defaults; the missing historical transform cannot be inferred from saved outputs.

The existing classifier structure, optimizer, schedule, epoch count, checkpoint path, and test-set selection behaviour are retained. `saved_models/weights.h5` is written with `torch.save`; despite its extension, it is a PyTorch state dictionary rather than an HDF5 model.

## Validated environment

The offline checks passed with the following locally installed environment:

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.1 |
| Pillow | 12.3.0 |
| tqdm | 4.70.0 |
| nbformat | 5.11.1 |

The requirements files pin these directly tested packages. JupyterLab is a separately included notebook interface and was not part of the command-line execution check. This is a maintenance environment, not the original training environment or a complete transitive dependency lock. CUDA binaries and hardware compatibility must be selected for the machine used for a full experiment.

## Offline check scope

From the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The structural check validates notebook format, checks that working outputs are empty, compiles Python code cells, and compares all three archived files to their recorded hashes.

The execution check runs the actual ResNet-50 code cells in a temporary directory. It substitutes generated CIFAR-shaped images, uses 16 training and 11 test samples with batch size 8, runs one epoch on CPU with randomly initialized weights, and blocks network connections. The default learning rate is retained. An uneven test split of 8 + 3 samples checks correct sample-weighted mean loss.

Assertions cover dataset split flags, saved checkpoint creation, 100-class model output, final evaluation count, loss averaging, five families with five evaluation levels each, and distinct Gaussian, salt-and-pepper, and mask strengths. Seeds are fixed for the fixture and severity checks. Temporary images and weights are not committed.

The three cells tagged `visualization-only` are skipped by this execution check; they are included in the structural compilation check. Complete CIFAR-100 training, downloaded pretrained weights, CUDA execution, interactive figure rendering, and execution of the other two notebooks were not validated. The fixture's predictions and accuracy are not useful model-performance measurements.

## Result interpretation

The original ViT notebook records **89.15%** accuracy after selecting a checkpoint by test-set accuracy. That value is retained as historical evidence, with test-set reuse stated explicitly.

The original ResNet-50 output reports **98.59%** over **50,000** images even though its dataset cell declares a **10,000**-image test split. Because its loader also uses an undefined variable, the archived notebook cannot establish which historical kernel state generated that number. The maintained copy clears this output; no replacement accuracy is claimed.

The ResNet-50 workflow saves the model with minimum test loss, but its final clean-image and perturbation evaluations use the model left in memory after the last epoch. The two models need not be the same. The configurable ResNet and ViT notebooks likewise evaluate the test split while selecting models. Before reporting a new benchmark, introduce a validation split from training data and evaluate the test split only after selection.

The configurable ResNet and ViT notebooks average per-batch losses rather than weighting them by batch size; a final partial batch can therefore affect the reported mean differently. Their original training code is preserved, not brought under the ResNet-50 execution guarantee. Preprocessing, architecture, initialization, and training duration vary across all three notebooks.

## Perturbation protocol

| Family | Parameter at level `L`, from 1 to 5 |
| --- | --- |
| Gaussian | Standard deviation `0.02 × L` |
| Salt and pepper | Severity `0.05 × L`, with randomly sampled channel coordinates |
| Brightness | Factor `0.75 + 0.25 × L` |
| Contrast | Factor `0.75 + 0.25 × L` |
| Mask | 8×8 patches; masked fraction `0.15 × L`, rounded down to a whole patch count |

Perturbations are applied to tensors before normalization. Brightness and contrast level 1 use factor 1.0, so there are 25 evaluation conditions rather than 25 distinct corruptions. Salt-and-pepper sampling can revisit coordinates and changes individual channels; its parameter is not an exact fraction of whole RGB pixels changed. Random corruptions are regenerated as the dataset is accessed, and the notebook does not archive fixed perturbed datasets or per-image random states.

## Continuing the experiments

Record the code revision, environment, hardware, random seeds, split definition, preprocessing, and all hyperparameters for each new run. Identify whether metrics come from the last epoch or a reloaded selected checkpoint. Keep raw metric files and executed notebooks with a run identifier so that future results can be compared to their actual settings. Dataset files, historical checkpoints, and a definitive original training environment are not included in this repository.

[Return to the project](../README.md).
