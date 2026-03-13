# ROCO510/ROCO510Z Coursework 1: Section 2 Report

## Title

Design, Implementation, and Evaluation of a Fashion-MNIST Classifier Using PyTorch

## Abstract

This report describes the design and evaluation of an image classification model for the Fashion-MNIST dataset. The work implements a reproducible PyTorch pipeline that performs dataset loading and preprocessing, train/validation/test splitting, model training with regularisation, and final evaluation on a held-out test set. A compact convolutional neural network (CNN) with batch normalisation and dropout is trained using AdamW optimisation and a cosine learning-rate schedule. Model performance is assessed using accuracy, confusion matrix, and per-class precision/recall/F1-score. In this repository, a 10-epoch CPU run achieved 92.33% test accuracy, exceeding the coursework target of 80% and demonstrating that the selected architecture and optimisation scheme is effective.

## 1. Introduction

Fashion-MNIST is a standard benchmark dataset of 28x28 grayscale images representing 10 clothing categories. Compared to MNIST digits, Fashion-MNIST exhibits higher intra-class variation and more ambiguous inter-class boundaries (e.g., shirt vs T-shirt/top), making it a suitable testbed for evaluating convolutional architectures and training procedures. The objective of this section is to build a classifier that exceeds 80% accuracy on the test set, while providing a defensible methodology and clear evaluation.

## 2. Data Loading and Preprocessing

### 2.1 Dataset

The dataset is loaded via `torchvision.datasets.FashionMNIST`, with the original split:

- Training set: 60,000 images
- Test set: 10,000 images

### 2.2 Train/Validation Split

To support model selection without leaking information from the test set, the training set is split into:

- Training subset: 50,000 images
- Validation subset: 10,000 images (default; configurable via `--val-size`)

The split is performed using `random_split` with a fixed random seed to ensure reproducibility.

### 2.3 Normalisation

Inputs are converted to tensors and normalised using standard Fashion-MNIST statistics:

- mean = 0.2860
- std = 0.3530

Normalisation reduces scale variation and improves optimisation stability, particularly when using adaptive optimisers.

### 2.4 Data Augmentation

Optional lightweight augmentation (`--augment`) applies a small random affine transform (rotation, translation, mild scale). This is intended to increase robustness to minor viewpoint changes and local misalignment, without violating label semantics.

## 3. Model Design and Implementation

### 3.1 Architecture

A compact CNN is used, motivated by the strong inductive bias of convolutional filters for image data (local connectivity and translation equivariance). The implemented model comprises:

- Three convolutional blocks with BatchNorm and ReLU activations
- Max pooling after each block to reduce spatial resolution
- A fully connected head with dropout regularisation

Batch normalisation improves training stability and permits higher learning rates. Dropout reduces co-adaptation in the dense layer and mitigates overfitting.

### 3.2 Loss Function

Multi-class classification uses cross-entropy loss.

### 3.3 Optimiser and Learning Rate

The model is trained with AdamW, which decouples weight decay from the gradient update and is commonly effective for CNNs. A cosine annealing learning-rate schedule is used across epochs to improve convergence.

## 4. Training Procedure

Training proceeds for a fixed number of epochs (`--epochs`) with mini-batches (`--batch-size`). At each epoch:

1. The model is trained on the training subset.
1. The model is evaluated on the validation subset.
1. The best checkpoint is selected by validation accuracy and saved to disk.

The script records training and validation loss/accuracy and produces a curves plot (`curves.png`) for analysis of convergence and overfitting.

## 5. Evaluation and Results

### 5.1 Metrics

Final performance is evaluated on the test set using:

- Accuracy
- Confusion matrix
- Per-class precision, recall, and F1-score (with macro and weighted averages)

These metrics provide more insight than accuracy alone, particularly where classes are frequently confused.

### 5.2 Empirical Result (Smoke Run)

On the provided environment (CPU execution), the following runs were observed:

- 1 epoch (smoke run): test accuracy 0.876
- 10 epochs: validation accuracy 0.9163, test accuracy 0.9233

Both results surpass the 0.80 requirement. The 10-epoch run demonstrates improved convergence relative to the smoke run, consistent with the expected learning dynamics of CNNs on Fashion-MNIST.

All evaluation outputs are saved per run to a timestamped folder under `runs/`, including `metrics.json`, `confusion_matrix.png`, and the best model checkpoint.

## 6. Discussion

The achieved accuracy after a minimal training run suggests that the selected architecture has sufficient capacity and inductive bias for Fashion-MNIST, while remaining lightweight and explainable. The primary residual errors are expected to occur among visually similar categories. Inspection of the confusion matrix provides a direct mechanism to identify these failure modes, which can motivate improvements such as stronger augmentation, tuned regularisation, or slightly deeper networks.

Limitations include:

- Performance depends on hyperparameters (learning rate, dropout, weight decay); tuning may be required across environments.
- If training is performed without augmentation, the model may overfit sooner, visible as a divergence between training and validation curves.
- GPU acceleration is not available in the current environment due to driver compatibility; training is therefore CPU-bound.

## 7. Conclusion

This section delivered a reproducible PyTorch pipeline for Fashion-MNIST classification, including preprocessing, a CNN architecture, training with regularisation, and detailed evaluation. The implementation meets the coursework requirement of achieving test accuracy above 80% and provides the additional diagnostic outputs necessary for a rigorous report (curves, confusion matrix, and per-class PRF metrics).

## References

1. Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
1. Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.
1. Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
