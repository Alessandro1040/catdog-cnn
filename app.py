#!/usr/bin/env python3
"""
app.py — demo web locale del classificatore gatto/cane.

Avvia un piccolo server Flask con una pagina dove trascini una foto (o la
scatti con la webcam) e vedi la predizione del modello con la confidenza.

Uso:
    python3 app.py                        # http://127.0.0.1:7860
    python3 app.py --port 8000            # porta diversa
    python3 app.py --host 0.0.0.0         # visibile anche dal telefono in rete locale
    python3 app.py --ckpt best_catdog.pt  # checkpoint alternativo

Nessuna dipendenza extra oltre a flask (requirements.txt).
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image, ImageFile, UnidentifiedImageError

from predict import eval_transform_for
from train import load_model, pick_device

ImageFile.LOAD_TRUNCATED_IMAGES = True     # il dataset Kaggle ha jpg troncati

HERE = Path(__file__).resolve().parent
SAMPLES_DIR = HERE / "samples"             # 6 foto di esempio versionate nel repo
MAX_UPLOAD_MB = 12
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# riempiti in main()
MODEL = None
DEVICE = None
TRANSFORM = None
CLASSES = None
CKPT_PATH = None
IMG_SIZE = None


def classify(img: Image.Image):
    """Ritorna (label, confidenza, probabilita' grezza, emoji) per una PIL.Image."""
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(MODEL(x)).item()      # 1 = Dog, 0 = Cat

    label = CLASSES[1] if prob > 0.5 else CLASSES[0]
    conf = prob if prob > 0.5 else 1.0 - prob
    emoji = "🐶" if label == CLASSES[1] else "🐱"
    return label, conf, prob, emoji


def list_samples():
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in ALLOWED_EXT)


PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gatto o Cane? — CNN da zero in PyTorch</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; padding: 32px 16px 64px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 20% -10%, #2b2a5e 0%, #12121c 55%, #0b0b12 100%);
    color: #eceaf6; display: flex; flex-direction: column; align-items: center;
  }
  h1 { margin: 0 0 6px; font-size: clamp(24px, 4vw, 38px); letter-spacing: -0.5px; text-align: center; }
  h1 span { background: linear-gradient(90deg, #8b7bff, #4fd1c5); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .sub { margin: 0 0 26px; color: #a9a5c7; font-size: 14px; text-align: center; max-width: 640px; line-height: 1.55; }
  .card {
    width: 100%; max-width: 720px; background: #171727ee; border: 1px solid #2e2d4a;
    border-radius: 18px; padding: 22px; box-shadow: 0 24px 60px rgba(0,0,0,.45);
  }
  .drop {
    border: 2px dashed #3d3b63; border-radius: 14px; padding: 30px 18px; text-align: center;
    transition: border-color .15s, background .15s; cursor: pointer; background: #12121d;
  }
  .drop.hot { border-color: #8b7bff; background: #1c1b30; }
  .drop p { margin: 8px 0 0; color: #9c98bd; font-size: 14px; }
  .drop b { color: #eceaf6; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 16px; }
  button, .btn {
    appearance: none; border: 1px solid #3d3b63; background: #22213a; color: #eceaf6;
    border-radius: 10px; padding: 10px 16px; font-size: 14px; cursor: pointer; font-weight: 600;
    text-decoration: none; display: inline-flex; align-items: center; gap: 8px;
  }
  button:hover, .btn:hover { background: #2c2b4a; }
  button.primary { background: linear-gradient(90deg, #6c5ce7, #4fd1c5); border: 0; color: #0b0b12; }
  button.primary:hover { filter: brightness(1.08); }
  button[disabled] { opacity: .5; cursor: progress; }
  #preview { margin-top: 18px; text-align: center; display: none; }
  #preview img { max-width: 100%; max-height: 320px; border-radius: 12px; border: 1px solid #2e2d4a; }
  #result { margin-top: 18px; display: none; }
  .verdict { display: flex; align-items: center; gap: 14px; padding: 16px; border-radius: 14px; background: #12121d; border: 1px solid #2e2d4a; }
  .verdict .emoji { font-size: 42px; line-height: 1; }
  .verdict .name { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
  .verdict .meta { font-size: 13px; color: #9c98bd; margin-top: 2px; }
  .bar { height: 10px; border-radius: 999px; background: #26243f; overflow: hidden; margin-top: 14px; }
  .bar > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, #6c5ce7, #4fd1c5); transition: width .35s ease; }
  .legend { display: flex; justify-content: space-between; font-size: 12px; color: #85809f; margin-top: 6px; }
  .samples { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 16px; }
  .samples img { width: 66px; height: 66px; object-fit: cover; border-radius: 10px; border: 1px solid #3d3b63; cursor: pointer; transition: transform .12s; }
  .samples img:hover { transform: translateY(-2px) scale(1.04); border-color: #8b7bff; }
  .err { color: #ff8a8a; font-size: 14px; margin-top: 14px; text-align: center; }
  footer { margin-top: 22px; color: #6f6b8c; font-size: 12px; text-align: center; line-height: 1.7; }
  code { background: #26243f; padding: 2px 6px; border-radius: 6px; font-size: 12px; }
</style>
</head>
<body>
  <h1>Gatto o <span>Cane</span>?</h1>
  <p class="sub">CNN addestrata <b>da zero</b> su 24.970 foto del dataset Dogs vs Cats &mdash; 96,45% di accuracy su 3.746 immagini di test. Trascina una foto per provarla.</p>

  <div class="card">
    <div class="drop" id="drop">
      <div style="font-size:34px">📷</div>
      <p><b>Trascina una foto</b> oppure clicca per sceglierla</p>
      <p style="margin-top:4px">JPG / PNG / WEBP &mdash; max 12 MB</p>
    </div>

    <div class="row">
      <button class="primary" id="pick">Scegli un file</button>
      <button id="camera">Usa la webcam</button>
      <button id="random">Foto a caso dagli esempi</button>
    </div>

    <input type="file" id="file" accept="image/*" hidden>
    <input type="file" id="cam" accept="image/*" capture="environment" hidden>

    <div id="preview"><img id="preview-img" alt="anteprima"></div>

    <div id="result">
      <div class="verdict">
        <div class="emoji" id="emoji">🐱</div>
        <div>
          <div class="name" id="label">&mdash;</div>
          <div class="meta" id="meta"></div>
        </div>
      </div>
      <div class="bar"><i id="bar"></i></div>
      <div class="legend"><span>Gatto 0%</span><span id="conf">&mdash;</span><span>Cane 100%</span></div>
    </div>

    <div class="samples" id="samples"></div>
    <div class="err" id="err"></div>
  </div>

  <footer>
    __INFO__<br>
    Modello: 4&times; ConvBlock + GlobalAvgPool + Dropout &middot; 1.206.241 parametri
  </footer>

<script>
const $ = (id) => document.getElementById(id);
const fileInput = $("file"), drop = $("drop");

function showError(msg) { $("err").textContent = msg || ""; }

function setPreview(src) {
  $("preview-img").src = src;
  $("preview").style.display = "block";
}

async function send(blob, name) {
  const fd = new FormData();
  fd.append("image", blob, name || "foto.jpg");
  const r = await fetch("/api/predict", { method: "POST", body: fd });
  const data = await r.json();
  if (!r.ok) { showError(data.error || "errore"); return null; }
  return data;
}

function render(d) {
  $("result").style.display = "block";
  $("emoji").textContent = d.emoji;
  $("label").textContent = d.label === "Dog" ? "CANE" : "GATTO";
  $("meta").textContent = "confidenza " + d.confidence_pct + "  ·  " + d.width + "x" + d.height + " px";
  $("bar").style.width = (d.prob_dog * 100).toFixed(1) + "%";
  $("conf").textContent = "sigmoide = " + d.prob_dog.toFixed(3);
}

function lock(on) {
  for (const b of document.querySelectorAll("button")) b.disabled = on;
}

async function handle(file) {
  if (!file) return;
  showError("");
  setPreview(URL.createObjectURL(file));
  $("result").style.display = "none";
  lock(true);
  try {
    const d = await send(file, file.name);
    if (d) render(d);
  } catch (e) {
    showError("richiesta fallita: " + e.message);
  } finally {
    lock(false);
  }
}

$("pick").onclick = () => fileInput.click();
$("camera").onclick = () => $("cam").click();
drop.onclick = () => fileInput.click();
fileInput.onchange = (e) => handle(e.target.files[0]);
$("cam").onchange = (e) => handle(e.target.files[0]);

drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hot"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("hot");
  handle(e.dataTransfer.files[0]);
});

async function tryUrl(url, name) {
  try {
    const r = await fetch(url);
    if (!r.ok) { showError("campione non disponibile"); return; }
    const blob = await r.blob();
    await handle(new File([blob], name || "campione.jpg", { type: blob.type }));
  } catch (e) { showError("richiesta fallita: " + e.message); }
}

$("random").onclick = async () => {
  showError("");
  const r = await fetch("/api/random-sample");
  const info = await r.json();
  if (!r.ok) { showError(info.error || "nessun campione"); return; }
  await tryUrl(info.url, info.name);
};

fetch("/api/samples").then(r => r.json()).then(d => {
  const box = $("samples");
  if (!d.samples || !d.samples.length) { box.style.display = "none"; return; }
  for (const name of d.samples) {
    const im = document.createElement("img");
    im.src = "/api/sample/" + name;
    im.title = "prova con " + name;
    im.onclick = () => tryUrl(im.src, name);
    box.appendChild(im);
  }
});
</script>
</body>
</html>

"""



@app.post("/api/predict")
def api_predict():
    """Accetta un file (campo `image`) e ritorna la predizione in JSON."""
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="nessun file ricevuto (campo 'image')"), 400

    ext = Path(f.filename).suffix.lower()
    if ext and ext not in ALLOWED_EXT:
        return jsonify(error=f"estensione non supportata: {ext}"), 400

    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return jsonify(error=f"immagine non leggibile: {e}"), 400

    label, conf, prob, emoji = classify(img)
    return jsonify(
        label=label,
        emoji=emoji,
        confidence=round(conf, 6),
        confidence_pct=f"{conf:.2%}",
        prob_dog=round(prob, 6),
        filename=f.filename,
        width=img.width,
        height=img.height,
        device=str(DEVICE),
        img_size=IMG_SIZE,
    )


@app.get("/api/samples")
def api_samples():
    return jsonify(samples=list_samples())


@app.get("/api/sample/<path:name>")
def api_sample(name: str):
    """Serve una foto di esempio (solo da samples/, niente path traversal)."""
    if name not in list_samples():
        return jsonify(error="campione non trovato"), 404
    return send_file(SAMPLES_DIR / name)


@app.get("/api/random-sample")
def api_random_sample():
    names = list_samples()
    if not names:
        return jsonify(error="cartella samples/ vuota"), 404
    name = random.choice(names)
    return jsonify(name=name, url=f"/api/sample/{name}")


@app.get("/")
def index():
    info = (f"device <code>{DEVICE}</code> &middot; input <code>{IMG_SIZE}x{IMG_SIZE}</code> "
            f"&middot; checkpoint <code>{CKPT_PATH.name}</code>")
    return Response(PAGE.replace("__INFO__", info), mimetype="text/html")


def main() -> int:
    global MODEL, DEVICE, TRANSFORM, CLASSES, CKPT_PATH, IMG_SIZE

    ap = argparse.ArgumentParser(description="Demo web locale del classificatore gatto/cane")
    ap.add_argument("--ckpt", type=Path, default=HERE / "best_catdog.pt")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = solo questo Mac, 0.0.0.0 = raggiungibile anche dal telefono")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.ckpt.is_file():
        print(f"Checkpoint non trovato: {args.ckpt}")
        print("Addestra il modello con `python3 train.py` oppure passa --ckpt <percorso>.")
        return 1

    DEVICE = pick_device()
    MODEL, IMG_SIZE, CLASSES = load_model(args.ckpt, DEVICE)
    TRANSFORM = eval_transform_for(IMG_SIZE)
    CKPT_PATH = args.ckpt

    print(f"Checkpoint : {args.ckpt.name}")
    print(f"Device     : {DEVICE}")
    print(f"Input      : {IMG_SIZE}x{IMG_SIZE} px — classi {CLASSES}")
    print(f"Campioni   : {len(list_samples())} immagini in samples/")
    print(f"\n  Demo pronta su http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    print("  (Ctrl+C per uscire)\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
