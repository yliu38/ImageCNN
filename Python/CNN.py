from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

###########################################################################
# Reproducibility 
###########################################################################
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # slower, deterministic

###########################################################################
# CLI
###########################################################################

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Image‑classification trainer")
    p.add_argument("--data_dir", required=True, type=Path, help="Root directory with class sub‑folders")
    p.add_argument("--arch", default="resnet18", help="torchvision.model architecture")
    p.add_argument("--epochs", default=25, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--lr", default=3e‑4, type=float)
    p.add_argument("--patience", default=5, type=int, help="Early‑stopping patience (epochs)")
    p.add_argument("--output_dir", default=Path("outputs"), type=Path)
    return p.parse_args()

###########################################################################
# Data
###########################################################################

def build_dataloaders(
    root: Path,
    batch: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[Dict[str, DataLoader], int]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_ds = datasets.ImageFolder(root, transform=train_tfms)
    n_cls = len(full_ds.classes)

    n_total = len(full_ds)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED)
    )
    val_ds.dataset.transform = eval_tfms
    test_ds.dataset.transform = eval_tfms

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4),
        "val": DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4),
        "test": DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=4),
    }
    return loaders, n_cls

###########################################################################
# Model
###########################################################################

def build_model(arch: str, num_classes: int) -> nn.Module:
    model_fn = getattr(models, arch)
    model: nn.Module = model_fn(weights="DEFAULT")

    if hasattr(model, "fc"):  # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):  # VGG, DenseNet
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

###########################################################################
# Training helpers
###########################################################################

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[np.ndarray] = []
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        running += criterion(out, y).item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_prob.extend(torch.softmax(out, dim=1).cpu().numpy())

    y_prob_arr = np.vstack(y_prob)
    y_pred = np.argmax(y_prob_arr, 1)

    metrics: Dict[str, float] = {
        "loss": running / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if y_prob_arr.shape[1] == 2:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob_arr[:, 1])
        except ValueError:
            metrics["auroc"] = float("nan")
    return metrics


def should_stop(epoch: int, best_epoch: int, patience: int) -> bool:
    return (epoch - best_epoch) >= patience

###########################################################################
# Visualisation
###########################################################################

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True", xlabel="Predicted", title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

###########################################################################
# Main
###########################################################################

def main() -> None:
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, n_cls = build_dataloaders(args.data_dir, args.batch_size)
    model = build_model(args.arch, n_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_epoch, best_val_loss = -1, float("inf")
    best_weights = None
    history: Dict[str, List[float]] = defaultdict(list)  # type: ignore

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        for k, v in val_metrics.items():
            history.setdefault(f"val_{k}", []).append(v)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} | val loss {val_metrics['loss']:.4f} | "
              f"val acc {val_metrics['accuracy']:.4f} | {elapsed:.1f}s")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_weights = model.state_dict()
            torch.save(best_weights, args.output_dir / "best_model.pt")

        if should_stop(epoch, best_epoch, args.patience):
            print("Early stopping – no improvement in", args.patience, "epochs")
            break

    # Restore best
    assert best_weights is not None, "Training loop did not store best weights"
    model.load_state_dict(best_weights)

    # Final evaluation on test set
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    # Confusion matrix
    y_true, y_pred = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loaders["test"]:
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy())
            y_pred.extend(torch.argmax(logits, 1).cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, loaders["train"].dataset.dataset.classes, args.output_dir / "confusion_matrix.png")

    # Persist metrics & training history
    with open(args.output_dir / "test_metrics.json", "w") as fp:
        json.dump(test_metrics, fp, indent=2)
    with open(args.output_dir / "history.json", "w") as fp:
        json.dump(history, fp, indent=2)

    # Save entire model for easy inference
    torch.save({"arch": args.arch, "state_dict": best_weights, "classes": loaders["train"].dataset.dataset.classes},
               args.output_dir / "model_full.pth")

    print("Artifacts saved to", args.output_dir.resolve())

###########################################################################
if __name__ == "__main__":
    main()
