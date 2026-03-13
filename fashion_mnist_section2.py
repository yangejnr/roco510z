#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested in ("cpu",):
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        return torch.device("cuda")
    if requested not in ("auto", ""):
        raise SystemExit("--device must be one of: auto, cpu, cuda")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


@torch.no_grad()
def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def prf_from_confusion(cm: np.ndarray) -> dict:
    # Per-class precision, recall, f1 + macro averages.
    num_classes = cm.shape[0]
    eps = 1e-12
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2.0 * prec * rec / (prec + rec + eps)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(int(cm[c, :].sum()))

    macro = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }
    weighted = {
        "precision": float(np.average(precisions, weights=supports)),
        "recall": float(np.average(recalls, weights=supports)),
        "f1": float(np.average(f1s, weights=supports)),
    }
    return {
        "per_class": [
            {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for p, r, f, s in zip(precisions, recalls, f1s, supports)
        ],
        "macro_avg": macro,
        "weighted_avg": weighted,
    }


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 14x14
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 7x7
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 3x3
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_curve_plot(history: list[EpochMetrics], out_path: Path) -> None:
    epochs = [m.epoch for m in history]
    train_loss = [m.train_loss for m in history]
    val_loss = [m.val_loss for m in history]
    train_acc = [m.train_acc for m in history]
    val_acc = [m.val_acc for m in history]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label="train")
    ax[1].plot(epochs, val_acc, label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Light annotation (avoid too much clutter).
    maxv = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=7, color=("white" if v > 0.6 * maxv else "black"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_fashion_mnist(
    data_dir: Path,
    *,
    augment: bool,
    seed: int,
    val_size: int,
) -> tuple[Dataset, Dataset, Dataset]:
    # Fashion-MNIST is 28x28 grayscale.
    # Normalisation constants are commonly used for Fashion-MNIST.
    mean = (0.2860,)
    std = (0.3530,)

    train_tfms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:
        train_tfms = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    train_full = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=transforms.Compose(train_tfms))
    test = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))

    if val_size <= 0 or val_size >= len(train_full):
        raise SystemExit(f"--val-size must be in (0, {len(train_full)})")

    train_size = len(train_full) - val_size
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(train_full, [train_size, val_size], generator=g)
    return train, val, test


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(1) == yb).sum().item())
        total += int(xb.size(0))

    return total_loss / max(1, total), total_correct / max(1, total)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(1) == yb).sum().item())
        total += int(xb.size(0))
    return total_loss / max(1, total), total_correct / max(1, total)


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p = logits.argmax(1).detach().cpu().numpy()
        preds.append(p)
        targets.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="ROCO510 Coursework 1 - Section 2 (Fashion-MNIST)")
    ap.add_argument("--data-dir", default="data", help="Where to download/store Fashion-MNIST")
    ap.add_argument("--run-dir", default="runs", help="Directory to store run artifacts")
    ap.add_argument("--tag", default="", help="Optional run tag (added to the run folder name)")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--val-size", type=int, default=10000)
    ap.add_argument("--augment", action="store_true", help="Enable lightweight data augmentation")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP (only relevant on CUDA)")
    ap.add_argument("--device", default="auto", help="Device selection: auto|cpu|cuda")

    args = ap.parse_args()

    set_seed(int(args.seed))
    device = pick_device(str(args.device))
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    run_root = Path(args.run_dir)
    ensure_dir(run_root)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = run_root / f"fashion_mnist_{stamp}{tag}"
    ensure_dir(out_dir)

    data_dir = Path(args.data_dir)
    ensure_dir(data_dir)

    train_ds, val_ds, test_ds = load_fashion_mnist(data_dir, augment=bool(args.augment), seed=int(args.seed), val_size=int(args.val_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleCNN(num_classes=10, dropout=float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.epochs)))

    history: list[EpochMetrics] = []
    best_val_acc = -math.inf
    best_path = out_dir / "best.pt"

    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device, use_amp=use_amp, scaler=scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step()

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                train_acc=float(train_acc),
                val_loss=float(val_loss),
                val_acc=float(val_acc),
            )
        )

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc}, best_path)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={opt.param_groups[0]['lr']:.3e}"
        )

    # Load best checkpoint for test evaluation.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = eval_epoch(model, test_loader, device)
    preds, targets = predict_all(model, test_loader, device)
    cm = confusion_matrix(preds, targets, num_classes=10)
    prf = prf_from_confusion(cm)

    save_curve_plot(history, out_dir / "curves.png")
    save_confusion_matrix_plot(cm, FASHION_MNIST_CLASSES, out_dir / "confusion_matrix.png")

    metrics = {
        "device": str(device),
        "use_amp": bool(use_amp),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "val_size": int(args.val_size),
        "augment": bool(args.augment),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "prf": prf,
        "classes": FASHION_MNIST_CLASSES,
        "outputs": {
            "best_checkpoint": str(best_path.resolve()),
            "curves_png": str((out_dir / "curves.png").resolve()),
            "confusion_matrix_png": str((out_dir / "confusion_matrix.png").resolve()),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f} out={out_dir.resolve()}")
    if test_acc < 0.80:
        print("WARNING: test accuracy < 0.80; try --epochs 15, --augment, or tune --lr/--dropout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
