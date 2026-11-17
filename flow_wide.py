"""Landscape flow diagram for slides (same content as flow.py, 2 rows x 4)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
FILL = {"data": "#cde2fb", "pre": "#d4f0e4", "model": "#e1dcf7", "eval": "#f9d6d6", "viz": "#fde6c2"}
EDGE = {"data": "#2a78d6", "pre": "#1baf7a", "model": "#6250d6", "eval": "#e34948", "viz": "#eda100"}
steps = [
    ("data", "1  Raw MHEALTH logs", "10 subjects · 50 Hz\n23 channels + label", "class balance,\nraw signal plots"),
    ("pre", "2  Channel selection", "keep acc + gyro (15 ch)\ndrop ECG, magnetometer", None),
    ("pre", "3  Drop null class", "label 0 = ~72%\nof all samples", None),
    ("pre", "4  Windowing", "2 s = 100 samples\n50% overlap → 6,684", "per-subject intensity,\njog vs run overlap"),
    ("pre", "5  Split by subject", "LOSO: test 1 · val 1\ntrain 8 · ×10 folds", None),
    ("pre", "6  Normalise", "z-score per channel\nstats from TRAIN only", None),
    ("model", "7  CNN-LSTM", "+ rotation/scale/jitter aug\n+ |v| magnitude channels", "training curves"),
    ("eval", "8  Evaluate", "accuracy, macro-F1\nvs RF, CNN, LSTM", "confusion matrix,\nmodel comparison"),
]
fig, ax = plt.subplots(figsize=(16.8, 7.4)); fig.patch.set_facecolor(SURF)
ax.set_xlim(0, 16.8); ax.set_ylim(0, 7.4); ax.axis("off")
W, H, gx = 3.4, 1.55, 0.6
rows_y = [4.75, 1.25]
pos = []
for i, (k, t, s, v) in enumerate(steps):
    r, c = divmod(i, 4)
    if r == 1: c = 3 - c          # snake: second row runs right-to-left
    x = 0.3 + c * (W + gx); y = rows_y[r]
    pos.append((x, y))
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12", fc=FILL[k], ec=EDGE[k], lw=2))
    ax.text(x + 0.18, y + H - 0.35, t, fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(x + 0.18, y + 0.52, s, fontsize=12, color=INK2, va="center", linespacing=1.35)
    if v:
        ax.add_patch(FancyBboxPatch((x + 0.25, y - 1.05), W - 0.5, 0.75, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=FILL["viz"], ec=EDGE["viz"], lw=1.4, ls="--"))
        ax.text(x + W / 2, y - 0.67, v, fontsize=10.5, ha="center", va="center", color=INK)
        ax.annotate("", (x + W / 2, y - 0.3), (x + W / 2, y), arrowprops=dict(arrowstyle="-|>", color=EDGE["viz"], lw=1.2))
for i in range(len(steps) - 1):
    (x1, y1), (x2, y2) = pos[i], pos[i + 1]
    if y1 == y2:
        if x2 > x1: a, b = (x1 + W, y1 + H / 2), (x2, y2 + H / 2)
        else: a, b = (x1, y1 + H / 2), (x2 + W, y2 + H / 2)
    else:
        a, b = (x1 + W + 0.25, y1 + H / 2), (x2 + W + 0.25, y2 + H / 2)
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, connectionstyle="arc3,rad=-0.5"))
        continue
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2))
ax.text(0.3, 7.0, "Pre-processing → model → evaluation", fontsize=16, fontweight="bold", color=INK)
ax.text(15.7, 7.0, "dashed = visualisation produced", fontsize=11, color=EDGE["viz"], ha="right", fontweight="bold")
fig.tight_layout(); fig.savefig(Path(__file__).parent / "fig/00b_flow_wide.png", dpi=180)
