import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from ml.model import CrnnRecognizer
from ml.metrics import character_error_rate
from ml.data.manifest import read_manifest

ck = torch.load("artifacts/best.pt", map_location="cpu", weights_only=True)
charset = Path("data/charset.txt").read_text(encoding="utf-8").splitlines()
m = CrnnRecognizer(len(charset)); m.load_state_dict(ck["model_state"]); m.eval()
rows = read_manifest(Path("data/eval_manifest.csv"))[:120]

def decode(logits):
    out, prev = [], -1
    for i in logits.argmax(-1).tolist():
        if i != prev and i != 0: out.append(charset[i])
        prev = i
    return "".join(out)

cers, examples = [], []
with torch.no_grad():
    for r in rows:
        img = Image.open(r.image_path).convert("L")
        w, h = img.size
        nw = max(1, min(320, int(round(w * (32.0 / h)))))
        img = img.resize((nw, 32), Image.BILINEAR)
        canvas = Image.new("L", (320, 32), 255); canvas.paste(img, (0, 0))
        a = np.asarray(canvas, dtype=np.float32) / 255.0
        a = (a - 0.5) / 0.5
        logits = m(torch.from_numpy(a)[None, None]).permute(1, 0, 2)[0]
        pred = decode(logits)
        cers.append(character_error_rate(r.text, pred))
        if len(examples) < 3:
            examples.append((r.text, pred, round(cers[-1], 3)))

print("checkpoint val_cer field:", round(float(ck.get("val_cer", -1)), 4))
print("samples:", len(cers), "mean CER:", round(float(np.mean(cers)), 4))
for ref, pred, c in examples:
    print("---")
    print("ref :", ref)
    print("pred:", pred, "| cer", c)