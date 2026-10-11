#!/usr/bin/env python3
"""
train.py — Cats vs Dogs: CNN da zero con tutte le best practice.

Sommario delle sezioni:
  §0  Riproducibilita' (seed)
  §1  Config (argparse)
  §2  Trasformazioni (augmentation SOLO per train) + normalizzazione
  §3  Dataset + split 70/15/15 train/val/test
  §4  Modello (CNN con BatchNorm + Dropout + Global Average Pooling)
  §5  Early stopping con salvataggio del modello migliore
  §6  Train / eval utils
  §7  Training loop (AdamW + ReduceLROnPlateau)
  §8  Grafici (loss / accuracy / learning rate)
  §9  Valutazione finale su test
  §10 Predizione su una nuova immagine

Uso:
    python train.py                          # config di default (128px, batch 64, 40 epoche)
    python train.py --epochs 15 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: salva i grafici su file, nessuna finestra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True     # il dataset Kaggle ha ~2000 jpg troncati: non crashare

# =========================================================
# 0. RIPRODUCIBILITA'
# =========================================================
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """Usa la GPU Apple (MPS) se disponibile, poi CUDA, altrimenti CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================
# 1. CONFIG
# =========================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Addestra una CNN da zero su Cats vs Dogs")
    ap.add_argument("--data-dir", type=Path, default=Path("data/cats_vs_dogs"),
                    help="cartella con le sottocartelle Cat/ e Dog/")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="dove salvare modello, grafici e metriche")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience-es", type=int, default=7, help="pazienza early stopping")
    ap.add_argument("--patience-lr", type=int, default=3, help="pazienza ReduceLROnPlateau")
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4,
                    help="worker del DataLoader (0 su Windows)")
    ap.add_argument("--demo", action="store_true",
                    help="a fine training predice una foto a caso del test set")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


# =========================================================
# 2. TRASFORMAZIONI (augmentation SOLO per train)
# =========================================================
MEAN = [0.485, 0.456, 0.406]               # statistiche ImageNet: standard su foto naturali
STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, eval_transform


# =========================================================
# 3. DATASET + SPLIT
# =========================================================
def build_loaders(data_dir: Path, img_size: int, batch_size: int, workers: int, seed: int):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Cartella '{data_dir}' non trovata.\n"
            "Atteso:\n"
            f"  {data_dir}/Cat/*.jpg\n"
            f"  {data_dir}/Dog/*.jpg\n"
            "Esegui prima:  python prepare_dataset.py"
        )

    train_transform, eval_transform = build_transforms(img_size)
    base = datasets.ImageFolder(data_dir)
    print(f"Classi: {base.classes}  (0={base.classes[0]}, 1={base.classes[1]})")

    n = len(base)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)           # split riproducibile
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Due dataset con transform diversi, poi Subset sugli stessi indici
    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), val_idx)
    test_ds = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), test_idx)

    pin = torch.cuda.is_available()               # pin_memory serve solo su CUDA
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, list(base.classes), eval_transform


# =========================================================
# 4. MODELLO
# =========================================================
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),      # 128 -> 64
            ConvBlock(32, 64),     # 64  -> 32
            ConvBlock(64, 128),    # 32  -> 16
            ConvBlock(128, 256),   # 16  -> 8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)          # (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),                      # logit singolo (classificazione binaria)
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# =========================================================
# 5. EARLY STOPPING (salva il modello migliore)
# =========================================================
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, path: Path | str = "best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = Path(path)
        self.best = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module, extra: dict | None = None) -> None:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            torch.save(
                {"state_dict": model.state_dict(),
                 "val_loss": val_loss,
                 "classes": (extra or {}).get("classes"),
                 "img_size": (extra or {}).get("img_size"),
                 "mean": MEAN, "std": STD},
                self.path,
            )
            print(f"    [ok] val_loss migliorata ({val_loss:.4f}) — modello salvato")
        else:
            self.counter += 1
            print(f"    [--] nessun miglioramento ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.stop = True


# =========================================================
# 6. TRAIN / EVAL UTILS
# =========================================================
def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# =========================================================
# 7. TRAINING LOOP
# =========================================================
def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}  |  seed: {args.seed}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_model = args.out_dir / "best_catdog.pt"

    train_loader, val_loader, test_loader, classes, eval_transform = build_loaders(
        args.data_dir, args.img_size, args.batch_size, args.workers, args.seed
    )

    model = CatDogCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience_lr, min_lr=1e-6
    )
    early = EarlyStopping(patience=args.patience_es, path=best_model)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, train=False)

        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(history.keys(), [tr_loss, va_loss, tr_acc, va_acc, current_lr]):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
              f"lr={current_lr:.2e} | {time.time() - t0:.0f}s", flush=True)

        early(va_loss, model, {"classes": classes, "img_size": args.img_size})
        if early.stop:
            print(f"\nEarly stopping all'epoca {epoch}. Miglior val_loss: {early.best:.4f}")
            break

    # ripristina i pesi migliori
    model.load_state_dict(torch.load(best_model, map_location=device)["state_dict"])
    print("Pesi migliori ripristinati.")

    # §8 grafici
    plot_history(history, args.out_dir / "training_history.png")

    # §9 valutazione finale su test
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"\nTEST — loss={test_loss:.4f} | accuracy={test_acc:.4f}")

    metrics = {
        "device": str(device),
        "seed": args.seed,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": round(early.best, 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "params": n_params,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
        "classes": classes,
        "train_minutes": round((time.time() - t_start) / 60, 1),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Metriche salvate in {args.out_dir / 'metrics.json'}")

    return {"model": model, "device": device, "eval_transform": eval_transform,
            "classes": classes, "test_loader": test_loader, "metrics": metrics}


# =========================================================
# 8. GRAFICI
# =========================================================
def plot_history(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoca")
    ax[0].legend()

    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoca")
    ax[1].legend()

    ax[2].plot(history["lr"])
    ax[2].set_title("Learning rate")
    ax[2].set_xlabel("epoca")
    ax[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Grafici salvati in {out_path}")


# =========================================================
# 9/10. PREDIZIONE SU NUOVA IMMAGINE
# =========================================================
def load_model(ckpt_path: Path, device: torch.device):
    """Ricostruisce il modello da un checkpoint salvato da train.py."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CatDogCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt.get("img_size") or 128
    classes = ckpt.get("classes") or ["Cat", "Dog"]
    return model, img_size, classes


def predict(image_path, model, device, eval_transform, classes, true_label=None):
    img = Image.open(image_path).convert("RGB")
    x = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    # index 1 = seconda classe in ordine alfabetico (dog); index 0 = cat
    label = classes[1] if prob > 0.5 else classes[0]
    conf = prob if prob > 0.5 else 1 - prob
    emoji = "CANE" if label == classes[1] else "GATTO"
    extra = f" | vero: {true_label}" if true_label is not None else ""
    print(f"{image_path} -> {label.upper()} ({emoji}, confidenza {conf:.2%}){extra}")
    return label, conf


# =========================================================
# MAIN
# =========================================================
def main() -> int:
    args = parse_args()
    out = train(args)

    if args.demo:
        test_loader = out.get("test_loader")
        if test_loader is not None:
            ds = test_loader.dataset                      # Subset
            path, true_idx = ds.dataset.samples[random.Random(args.seed).choice(list(ds.indices))]
            print("\nPredizione di esempio (da una foto del test set):")
            predict(path, out["model"], out["device"], out["eval_transform"],
                    out["classes"], true_label=out["classes"][true_idx])

    print(f"\nDone — test accuracy {out['metrics']['test_accuracy']:.2%} "
          f"in {out['metrics']['train_minutes']} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
