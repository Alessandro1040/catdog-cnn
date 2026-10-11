#!/usr/bin/env python3
"""
prepare_dataset.py — scarica e prepara il dataset "Cats vs Dogs" (25.000 foto).

Cosa fa, in ordine:
  1. scarica kagglecatsanddogs_5340.zip (~825 MB) dal mirror pubblico Microsoft
     (stesso contenuto del dataset Kaggle originale) — se non e' gia' presente
  2. estrae PetImages/ dallo zip
  3. sistema le cartelle come richiesto dal training:
        data/cats_vs_dogs/Cat/*.jpg
        data/cats_vs_dogs/Dog/*.jpg
  4. rimuove le immagini corrotte / non leggibili (nel dataset Kaggle ce ne sono ~2000)
  5. stampa un riassunto con il conteggio finale per classe

Uso:
    python prepare_dataset.py
    python prepare_dataset.py --keep-zip            # non cancella lo zip dopo l'estrazione
    python prepare_dataset.py --limit 4000          # solo 2000 Cat + 2000 Dog (training veloce)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

URL = (
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
    "kagglecatsanddogs_5340.zip"
)
DEFAULT_ZIP = Path("data/kagglecatsanddogs_5340.zip")
DEFAULT_DEST = Path("data/cats_vs_dogs")
CLASSES = ("Cat", "Dog")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def download(url: str, zip_path: Path, min_size: int = 700 * 1024 * 1024) -> Path:
    """Scarica lo zip con progresso. Riprende un download parziale se esiste."""
    if zip_path.exists() and zip_path.stat().st_size >= min_size:
        print(f"[1/4] zip gia' presente: {zip_path} ({human(zip_path.stat().st_size)})")
        return zip_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    part = zip_path.with_name(zip_path.name + ".part")
    already = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "catdog-cnn/1.0"})
    if already:
        req.add_header("Range", f"bytes={already}-")
        print(f"[1/4] riprendo il download da {human(already)}")
    else:
        print(f"[1/4] scarico {url}")

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = resp.length or 0
        if resp.status == 206:
            total += already
        mode = "ab" if already and resp.status == 206 else "wb"
        done = already if mode == "ab" else 0
        t0 = time.time()
        next_tick = 0.0
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time() - t0
                if now >= next_tick:
                    next_tick = now + 5
                    speed = done / max(now, 1e-6)
                    pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                    eta = f"  eta {(total - done) / speed:5.0f}s" if total else ""
                    print(f"      {pct}  {human(done)} / {human(total)}  ({human(speed)}/s){eta}",
                          flush=True)

    os.replace(part, zip_path)
    print(f"      scaricato: {zip_path} ({human(zip_path.stat().st_size)})")
    return zip_path


def extract(zip_path: Path, dest: Path) -> None:
    """Estrae PetImages/{Cat,Dog} direttamente in dest/{Cat,Dog}."""
    if all((dest / c).is_dir() for c in CLASSES):
        print(f"[2/4] immagini gia' estratte in {dest}")
        return

    print(f"[2/4] estraggo {zip_path.name} (solo PetImages/) ...")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist()
            if m.startswith("PetImages/") and not m.endswith("/")
            and not os.path.basename(m).startswith(".")      # salta .DS_Store, Thumbs.db ecc.
        ]
        total = len(members)
        for m in members:
            rel = Path(m).relative_to("PetImages")           # es. Cat/0.jpg
            if rel.parts[0] not in CLASSES:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 256)
            n += 1
            if n % 2000 == 0 or n == total:
                print(f"      {n}/{total} file  ({time.time() - t0:.0f}s)", flush=True)
    print(f"      estratti {n} file in {time.time() - t0:.0f}s")


def clean(dest: Path) -> tuple[int, int]:
    """Rimuove file non-immagine, corrotti o duplicati. Ritorna (rimossi, tenuti)."""
    print("[3/4] controllo integrita' delle immagini (puo' richiedere qualche minuto) ...")
    removed, kept = 0, 0
    seen: set[tuple[str, int, int]] = set()
    for cls in CLASSES:
        folder = dest / cls
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir() if p.is_file())
        for i, p in enumerate(files, 1):
            reason = None
            if p.suffix.lower() not in IMG_EXT:
                reason = "estensione non immagine"
            elif p.stat().st_size == 0:
                reason = "file vuoto"
            else:
                try:
                    with Image.open(p) as im:
                        im.verify()                          # rileva i file troncati/corrotti
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        if im.getextrema() == ((0, 0), (0, 0), (0, 0)):
                            reason = "immagine completamente nera"
                except Exception:
                    reason = "immagine corrotta"
            if reason is None:
                kept += 1
            else:
                removed += 1
                if removed <= 10:
                    print(f"      rimuovo {p}  ({reason})")
                p.unlink()
            if i % 2500 == 0:
                print(f"      [{cls}] {i}/{len(files)} analizzati", flush=True)
        # duplicati esatti (stessa dimensione e stesso contenuto)
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            key = (cls, p.stat().st_size, hash(p.read_bytes()))
            if key in seen:
                p.unlink()
                removed += 1
            else:
                seen.add(key)
    return removed, kept


def limit(dest: Path, per_class: int, seed: int = 42) -> None:
    """Tiene solo `per_class` immagini per classe (utile per prove rapide)."""
    rng = random.Random(seed)
    for cls in CLASSES:
        files = sorted(p for p in (dest / cls).iterdir() if p.is_file())
        if 0 < per_class < len(files):
            keep = set(rng.sample(files, per_class))
            for p in files:
                if p not in keep:
                    p.unlink()
        print(f"      [{cls}] tenute {len(list((dest / cls).iterdir()))} immagini")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepara data/cats_vs_dogs per il training")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--keep-zip", action="store_true", help="non cancellare lo zip a fine lavoro")
    ap.add_argument("--skip-clean", action="store_true", help="salta la pulizia delle immagini corrotte")
    ap.add_argument("--limit", type=int, default=0, help="max immagini per classe (0 = tutte)")
    args = ap.parse_args()

    t0 = time.time()
    zip_path = download(args.url, args.zip)
    extract(zip_path, args.dest)
    if args.skip_clean:
        print("[3/4] pulizia saltata (--skip-clean)")
    else:
        removed, kept = clean(args.dest)
        print(f"      rimosse {removed} immagini non valide, {kept} valide")
    if args.limit:
        limit(args.dest, args.limit)

    print("[4/4] riepilogo")
    total = 0
    for cls in CLASSES:
        n = len([p for p in (args.dest / cls).iterdir() if p.is_file()])
        total += n
        print(f"      {args.dest / cls}: {n} immagini")
    print(f"      totale: {total} immagini in {time.time() - t0:.0f}s")

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"      zip rimosso ({zip_path})")

    if total == 0:
        print("ERRORE: nessuna immagine trovata.", file=sys.stderr)
        return 1
    print("\nOK — dataset pronto. Ora puoi lanciare:  python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
