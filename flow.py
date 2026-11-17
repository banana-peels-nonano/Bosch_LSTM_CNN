"""Pipeline flow diagram (required by the brief)."""
import matplotlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
FILL = {"data": "#cde2fb", "pre": "#d4f0e4", "viz": "#fde6c2", "model": "#e1dcf7", "eval": "#f9d6d6"}
EDGE = {"data": "#2a78d6", "pre": "#1baf7a", "viz": "#eda100", "model": "#6250d6", "eval": "#e34948"}

steps = [
    ("data", "Raw MHEALTH logs", "10 subjects · 50 Hz · 23 channels + label"),
    ("pre", "Channel selection", "keep acc + gyro = 15 ch\ndrop ECG, magnetometer"),
    ("pre", "Drop null class", "label 0 (≈72% of samples)"),
    ("pre", "Segment into windows", "2 s = 100 samples, 50% overlap\nonly inside one-label runs → 6,684"),
    ("pre", "Subject-wise split (LOSO)", "test 1 subject · val 1 · train 8\nrepeat for all 10 subjects"),
    ("pre", "Z-score per channel", "mean/std from TRAIN fold only"),
    ("model", "CNN-LSTM", "2× Conv1D(64,k5)+BN+ReLU → MaxPool\n→ LSTM(64) → Dropout → Dense(12)"),
    ("eval", "Evaluate", "accuracy, macro-F1, confusion matrix\nvs RF baseline, CNN, LSTM ablations"),
]
viz = [(0, "Class balance,\nraw signal plots"), (3, "Per-subject intensity,\njog vs run overlap"),
       (6, "Training curves"), (7, "Confusion matrix,\nper-subject scores")]

fig, ax = plt.subplots(figsize=(9, 12)); fig.patch.set_facecolor(SURF)
ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.45 + 0.4); ax.axis("off")
W, H, X0 = 5.2, 1.05, 0.4
ys = [len(steps) * 1.45 - 1.05 - i * 1.45 for i in range(len(steps))]
for (k, t, s), y in zip(steps, ys):
    ax.add_patch(FancyBboxPatch((X0, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[k], ec=EDGE[k], lw=1.6))
    ax.text(X0 + 0.2, y + H - 0.28, t, fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(X0 + 0.2, y + 0.36, s, fontsize=9, color=INK2, va="center", linespacing=1.3)
for y1, y2 in zip(ys[:-1], ys[1:]):
    ax.annotate("", (X0 + W/2, y2 + H), (X0 + W/2, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
for i, t in viz:
    y = ys[i]
    ax.add_patch(FancyBboxPatch((7.0, y + 0.12), 2.7, 0.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=FILL["viz"], ec=EDGE["viz"], lw=1.3))
    ax.text(8.35, y + 0.52, t, fontsize=8.8, ha="center", va="center", color=INK)
    ax.annotate("", (7.0, y + 0.52), (X0 + W, y + 0.52),
                arrowprops=dict(arrowstyle="-|>", color=EDGE["viz"], lw=1.2, ls="--"))
ax.text(8.35, ys[0] + H + 0.25, "Visualisation", ha="center", fontsize=11, fontweight="bold", color=INK)
ax.text(X0, ys[0] + H + 0.25, "Pre-processing & modelling pipeline", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(str(ROOT) + "/fig/00_flow_diagram.png", dpi=200)
