#!/usr/bin/env python3
"""
predict.py — classifica nuove immagini con il modello addestrato da train.py.

Uso:
    python predict.py mia_foto.jpg
    python predict.py foto1.jpg foto2.jpg foto3.jpg
    python predict.py mia_foto.jpg --ckpt best_catdog.pt --figure risultato.png

Il checkpoint `best_catdog.pt` contiene anche img_size, classi e statistiche di
normalizzazione, quindi la trasformazione di eval viene ricostruita da sola.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: nessuna finestra, solo file
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile
from torchvision import transforms

from train import MEAN, STD, load_model, pick_device

ImageFile.LOAD_TRUNCATED_IMAGES = True


def eval_transform_for(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def predict(image_path, model, device, transform, classes):
    """Ritorna (label, confidenza) per una singola immagine."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    label = classes[1] if prob > 0.5 else classes[0]     # 1 = Dog, 0 = Cat
    conf = prob if prob > 0.5 else 1 - prob
    return label, conf, img


def main() -> int:
    ap = argparse.ArgumentParser(description="Predizione gatto/cane su nuove immagini")
    ap.add_argument("images", nargs="+", type=Path, help="una o piu' immagini")
    ap.add_argument("--ckpt", type=Path, default=Path("best_catdog.pt"))
    ap.add_argument("--figure", type=Path, default=None,
                    help="se indicato, salva qui una griglia con le predizioni")
    args = ap.parse_args()

    device = pick_device()
    model, img_size, classes = load_model(args.ckpt, device)
    transform = eval_transform_for(img_size)
    print(f"Checkpoint: {args.ckpt} | device: {device} | img_size: {img_size} | classi: {classes}")

    grid, shown = [], []
    for p in args.images:
        if not p.is_file():
            print(f"salto (non trovata): {p}")
            continue
        label, conf, img = predict(p, model, device, transform, classes)
        emoji = "cane" if label == classes[1] else "gatto"
        print(f"{p.name} -> {label.upper()} ({emoji}) — confidenza {conf:.2%}")
        grid.append((img, label, conf))
        shown.append((str(p), label, round(conf, 4)))

    if args.figure and grid:
        fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 4), squeeze=False)
        for ax, (img, label, conf) in zip(axes[0], grid):
            ax.imshow(img)
            ax.set_title(f"{label} — {conf:.2%}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.figure, dpi=120)
        plt.close(fig)
        print(f"Figura salvata in {args.figure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
