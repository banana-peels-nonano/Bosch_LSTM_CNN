"""EDA figures: class balance, raw signals by activity, sensor separability."""
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_subject, ACTIVITIES, CH_NAMES, FS

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.titlecolor": INK, "axes.titleweight": "bold"})
OUT = str(ROOT) + "/fig"

# 1) class balance in windows
d = np.load(str(ROOT) + "/windows.npz"); y = d["y"]
cnt = np.bincount(y)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(range(12), cnt, color=BLUE, height=0.7)
ax.set_yticks(range(12), ACTIVITIES); ax.invert_yaxis()
for i, c in enumerate(cnt):
    ax.text(c + 8, i, str(c), va="center", color=INK2, fontsize=9)
ax.set_xlabel("2 s windows (all 10 subjects)")
ax.set_title("Windows per activity: balanced except Jump (only 20 reps recorded)", loc="left")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/01_class_balance.png", dpi=200); plt.close(fig)

# 2) raw signals: ankle acc magnitude, 4 s of each activity, subject 1
X, lab = load_subject(1)
ank = np.linalg.norm(X[:, 3:6], axis=1)
fig, axes = plt.subplots(3, 4, figsize=(12, 6.5), sharey=True)
t = np.arange(4 * FS) / FS
for k, ax in enumerate(axes.flat):
    idx = np.flatnonzero(lab == k + 1)
    seg = ank[idx[len(idx)//2: len(idx)//2 + 4*FS]]
    ax.plot(t[:len(seg)], seg, color=BLUE, lw=1.4)
    ax.set_title(ACTIVITIES[k], fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.6)
    if k % 4 == 0: ax.set_ylabel("|acc| m/s²")
    if k >= 8: ax.set_xlabel("time (s)")
fig.suptitle("Left-ankle acceleration magnitude, subject 1 (4 s per activity)",
             x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/02_raw_signals.png", dpi=200); plt.close(fig)

# 3) jogging vs running: why they get confused (per-window ankle |acc| std)
Xw, g = d["X"], d["g"]
m = np.linalg.norm(Xw[:, :, 3:6], axis=2)
fig, ax = plt.subplots(figsize=(8, 4))
for c, col, name in [(9, ORANGE, "Jogging"), (10, BLUE, "Running"), (3, AQUA, "Walking")]:
    sel = y == c
    ax.hist(m[sel].std(1), bins=40, alpha=0.55, color=col, label=name)
ax.set_xlabel("std of ankle |acc| within a 2 s window (m/s²)"); ax.set_ylabel("windows")
ax.set_title("Jogging and running overlap heavily in intensity, walking does not", loc="left")
ax.legend(frameon=False); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/03_jog_vs_run.png", dpi=200); plt.close(fig)

# 4) per-subject variation: mean ankle |acc| std for running, per subject
fig, ax = plt.subplots(figsize=(8, 3.6))
for c, col, name in [(9, ORANGE, "Jogging"), (10, BLUE, "Running")]:
    vals = [m[(y == c) & (g == s)].std(1).mean() for s in range(1, 11)]
    ax.plot(range(1, 11), vals, "o-", color=col, lw=2, ms=7, label=name)
ax.set_xticks(range(1, 11)); ax.set_xlabel("subject")
ax.set_ylabel("mean intensity (m/s²)")
ax.set_title("One person's 'jog' is another's 'run' → why we test on unseen subjects", loc="left")
ax.legend(frameon=False); ax.grid(color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT}/04_subject_variation.png", dpi=200); plt.close(fig)
print("ok")
