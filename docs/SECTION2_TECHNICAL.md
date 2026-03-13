# Section 2: Fashion-MNIST (Technical Documentation)

This document provides technical documentation for the Section 2 implementation in `fashion_mnist_section2.py`.

## 1. Overview

The script:

- Downloads and loads Fashion-MNIST using `torchvision.datasets.FashionMNIST`
- Splits the training set into train/validation using `torch.utils.data.random_split`
- Trains a small CNN classifier in PyTorch
- Logs per-epoch loss/accuracy for train and validation
- Evaluates on the test set and reports:
  - test accuracy
  - confusion matrix
  - per-class precision, recall, and F1
- Writes run artifacts (plots, metrics, checkpoint) under `runs/`

## 2. Dataset and Preprocessing

### 2.1 Dataset

Fashion-MNIST contains 28x28 grayscale images in 10 classes:

`T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot`

### 2.2 Normalisation

The implementation normalises tensors with typical Fashion-MNIST constants:

- mean = 0.2860
- std = 0.3530

Normalisation stabilises optimisation by bringing inputs to a standard scale, improving gradient conditioning and convergence behaviour.

### 2.3 Optional Data Augmentation

With `--augment`, the training transform includes:

- `RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))`

This is intentionally lightweight and label-preserving for clothing items, and can improve generalisation by making the model robust to small rotations and translations.

## 3. Train/Validation/Test Split

The original training set (60,000 images) is split into:

- train: `60000 - --val-size`
- validation: `--val-size` (default 10,000)

The split uses a fixed random seed (`--seed`) to ensure reproducibility.

The test set (10,000 images) is not used for training or model selection; it is evaluated only after training.

## 4. Model Architecture

The classifier is a compact CNN (`SimpleCNN`) suitable for 28x28 grayscale input:

- Conv(1->32) + BatchNorm + ReLU + MaxPool
- Conv(32->64) + BatchNorm + ReLU + MaxPool
- Conv(64->128) + BatchNorm + ReLU + MaxPool
- Fully connected head with Dropout:
  - Linear(128*3*3 -> 256) + ReLU + Dropout
  - Linear(256 -> 10)

Batch normalisation improves training stability. Dropout reduces overfitting in the dense head.

## 5. Optimisation

Training uses:

- Loss: cross-entropy
- Optimiser: AdamW (`--lr`, `--weight-decay`)
- Scheduler: cosine annealing over `--epochs`

This combination typically exceeds 80% test accuracy on Fashion-MNIST with modest tuning.

## 6. Evaluation Metrics

The script computes:

- accuracy for train/validation per epoch
- final test accuracy
- confusion matrix (10x10) on the test set
- per-class precision, recall, F1; plus macro and weighted averages

All results are saved to `metrics.json` for reproducible reporting.

## 7. Outputs (Run Artifacts)

Each run creates a timestamped directory under `runs/`, containing:

- `best.pt`: best checkpoint by validation accuracy
- `metrics.json`: configuration + best val accuracy + test accuracy + PRF + file paths
- `curves.png`: train vs validation loss/accuracy curves
- `confusion_matrix.png`: test confusion matrix heatmap

## 8. CLI Usage

Typical run:

```bash
. .venv/bin/activate
python fashion_mnist_section2.py --epochs 10 --batch-size 128 --augment --device cpu
```

CPU-only run:

```bash
python fashion_mnist_section2.py --device cpu --epochs 15 --augment
```

If the machine has a compatible NVIDIA driver, `--device auto` will pick CUDA; otherwise it falls back to CPU.

## 9. Reproducibility Note (This Repository)

In this environment, PyTorch is installed with CUDA support but CUDA is not available due to driver compatibility, so training runs on CPU. A 10-epoch run with the default CNN achieved `test_acc=0.9233` and saved artifacts under `runs/`.
