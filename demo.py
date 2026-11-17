"""Watch the final model classify a person it has never seen.

Uses the saved weights of the final model from LOSO fold 1
(trained on subjects 3-10, early-stopped on subject 2, never saw subject 1)
and streams subject 1's recording through it window by window.

  python demo.py                 # terminal stream + accuracy at the end
  python demo.py --plot          # live matplotlib animation (signal + predicted vs true)
  python demo.py --plot --speed 4
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data import load_subject, runs, ACTIVITIES, FS
from models import CNNLSTM
from train import add_mag

ap = argparse.ArgumentParser()
ap.add_argument("--plot", action="store_true")
ap.add_argument("--speed", type=float, default=8.0, help="x real time (plot mode)")
a = ap.parse_args()

# 1. normalisation stats: recomputed exactly as in training (train subjects 3..10 only)
D = np.load(ROOT / "windows.npz")
tr = add_mag(D["X"][(D["g"] != 1) & (D["g"] != 2)]).reshape(-1, 20)
mu, sd = tr.mean(0), tr.std(0) + 1e-6

# 2. model
model = CNNLSTM(c_in=20)
model.load_state_dict(torch.load(ROOT / "out/cnn_lstm_aug_mag_fold1.pt", map_location="cpu"))
model.eval()

# 3. stream subject 1: 2 s windows every 1 s, skipping null (label 0) segments
X, y = load_subject(1)
segs = []
for s, e, lab in runs(y):
    if lab == 0:
        continue
    for t in range(s, e - 100 + 1, 50):
        segs.append((t, lab - 1))

def predict(t):
    w = (add_mag(X[t:t + 100][None]) - mu) / sd
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(w.astype(np.float32))), 1)[0].numpy()
    return int(p.argmax()), float(p.max())

if not a.plot:
    correct = 0
    for i, (t, true) in enumerate(segs):
        pred, conf = predict(t)
        correct += pred == true
        mark = "ok " if pred == true else "XX "
        print(f"{mark} t={t/FS:7.1f}s  true={ACTIVITIES[true]:16s} pred={ACTIVITIES[pred]:16s} conf={conf:.2f}")
        time.sleep(0.02)
    print(f"\nsubject 1 (unseen): {correct}/{len(segs)} windows correct = {correct/len(segs)*100:.1f}%")
    sys.exit()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(figsize=(11, 4.5))
ank = np.linalg.norm(X[:, 3:6], axis=1)
line, = ax.plot([], [], color="#2a78d6", lw=1.2)
ax.set_ylim(0, 40); ax.set_xlim(0, 2); ax.set_xlabel("time in window (s)"); ax.set_ylabel("ankle |acc| m/s²")
txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=13, family="monospace")
score = [0, 0]

def step(i):
    t, true = segs[i]
    pred, conf = predict(t)
    score[0] += pred == true; score[1] += 1
    line.set_data(np.arange(100) / FS, ank[t:t + 100])
    col = "#0ca30c" if pred == true else "#d03b3b"
    txt.set_text(f"true: {ACTIVITIES[true]}\npred: {ACTIVITIES[pred]}  ({conf:.0%})\n"
                 f"running acc: {score[0]/score[1]:.1%}  [{score[1]}/{len(segs)}]")
    txt.set_color(col)
    return line, txt

anim = FuncAnimation(fig, step, frames=len(segs), interval=1000 / a.speed, blit=False, repeat=False)
plt.title("CNN-LSTM on subject 1 — never seen in training", loc="left", fontweight="bold")
plt.tight_layout(); plt.show()
