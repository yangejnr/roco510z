# Section 2: Fashion-MNIST Classification (Implementation)

This section implements the Fashion-MNIST classification task using PyTorch as described in the coursework brief:

- Load and preprocess Fashion-MNIST
- Split into train/validation/test
- Train a CNN model
- Plot training/validation loss and accuracy curves
- Evaluate on test set (accuracy + confusion matrix + precision/recall/F1)

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements_section2.txt
```

## Run

```bash
. .venv/bin/activate
python fashion_mnist_section2.py --epochs 10 --batch-size 128 --augment --device cpu
```

Artifacts are written under `runs/`, including:

- `metrics.json`
- `curves.png`
- `confusion_matrix.png`
- `best.pt`

## Notes

- The script targets >80% test accuracy; the included CNN exceeded this in practice (e.g., `test_acc=0.9233` after 10 epochs on CPU in this repo).
- If your machine supports CUDA, you can omit `--device cpu` and use `--device auto` (default).
