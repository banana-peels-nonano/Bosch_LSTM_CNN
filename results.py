"""Results figures + summary table from out/*.json."""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import ACTIVITIES

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.titlecolor": INK, "axes.titleweight": "bold"})
O, F = str(ROOT) + "/out", str(ROOT) + "/fig"
L = lambda n: json.load(open(f"{O}/{n}.json"))

order = [("rf_loso", "Random Forest\n(features)", YELLOW), ("lstm_loso", "LSTM only", MAGENTA),
         ("cnn_loso", "CNN only", AQUA), ("cnn_lstm_loso", "CNN-LSTM", ORANGE),
         ("cnn_lstm_aug_loso", "CNN-LSTM\n+ augmentation", BLUE),
         ("cnn_lstm_aug_mag_loso", "CNN-LSTM + aug\n+ magnitude ch.", "#6250d6")]
order = [o for o in order if os.path.exists(f"{O}/{o[0]}.json")]
R = {k: L(k) for k, _, _ in order}

# summary table
rows = []
for k, lab, _ in order:
    d = R[k]
    rows.append(f"| {lab.replace(chr(10),' ')} | {d['params'] or '-'} | {d['mean_acc']*100:.1f} ± {d['std_acc']*100:.1f} "
                f"| {d['mean_f1']*100:.1f} ± {d['std_f1']*100:.1f} | {min(f['acc'] for f in d['folds'])*100:.1f} |")
rand = {k: L(k) for k in ["cnn_lstm_random", "cnn_random"] if os.path.exists(f"{O}/{k}.json")}
with open(f"{O}/summary.md", "w") as fh:
    fh.write("| Model | Params | LOSO acc % (mean ± sd) | LOSO macro-F1 % | Worst subject acc % |\n|---|---|---|---|---|\n")
    fh.write("\n".join(rows) + "\n\nRandom window split (leaky):\n")
    for k, d in rand.items():
        fh.write(f"- {k}: acc {d['acc']*100:.2f}%, F1 {d['f1']*100:.2f}%\n")
print(open(f"{O}/summary.md").read())

# 1) model comparison: mean bar + per-subject dots
fig, ax = plt.subplots(figsize=(10.5, 4.8))
rng = np.random.default_rng(0)
for i, (k, lab, col) in enumerate(order):
    d = R[k]; accs = np.array([f["acc"] for f in d["folds"]]) * 100
    ax.bar(i, accs.mean(), color=col, width=0.6, alpha=0.9)
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(accs)), accs, s=22, color=INK, zorder=3, alpha=0.7)
    ax.text(i, accs.mean() + 1.2, f"{accs.mean():.1f}%", ha="center", color=INK, fontweight="bold")
ax.set_xticks(range(len(order)), [o[1] for o in order])
ax.set_ylim(50, 104); ax.set_ylabel("accuracy on unseen subject (%)")
ax.set_title("Leave-one-subject-out accuracy (bar = mean, dot = one test subject)", loc="left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{F}/10_model_comparison.png", dpi=200); plt.close(fig)

# 2) confusion matrix of the main model
main = "cnn_lstm_aug_mag_loso"
cm = np.array(R[main]["cm"]); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(8.5, 7.2))
ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#2a78d6", "#104281"]), vmin=0, vmax=1)
for i in range(12):
    for j in range(12):
        if cm[i, j]:
            ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.55 else INK)
ax.set_xticks(range(12), ACTIVITIES, rotation=45, ha="right"); ax.set_yticks(range(12), ACTIVITIES)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Confusion matrix, {main.replace('_loso','')} (row %, pooled over 10 LOSO folds)", loc="left", fontsize=10.5)
ax.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig(f"{F}/11_confusion_matrix.png", dpi=200); plt.close(fig)

# 3) training curves, fold where subject 1 is the test subject
h = R[main]["hist"]["1"]
ep = [x["epoch"] for x in h]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ep, [x["train_loss"] for x in h], color=BLUE, lw=2, label="train")
ax[0].plot(ep, [x["val_loss"] for x in h], color=ORANGE, lw=2, label="validation (unseen subject)")
ax[0].set_title("Loss", loc="left"); ax[0].set_xlabel("epoch"); ax[0].legend(frameon=False); ax[0].grid(color=GRID)
ax[1].plot(ep, [x["val_f1"] for x in h], color=ORANGE, lw=2)
b = int(np.argmax([x["val_f1"] for x in h]))
ax[1].scatter([ep[b]], [h[b]["val_f1"]], s=60, color=INK, zorder=3)
ax[1].annotate(f"best epoch {ep[b]} → weights kept", (ep[b], h[b]["val_f1"]), xytext=(8, -18),
               textcoords="offset points", color=INK2, fontsize=9)
ax[1].set_title("Validation macro-F1 (early stopping)", loc="left"); ax[1].set_xlabel("epoch"); ax[1].grid(color=GRID)
fig.tight_layout(); fig.savefig(f"{F}/12_training_curves.png", dpi=200); plt.close(fig)

# 4) leakage: random split vs LOSO
if "cnn_lstm_random" in rand:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    vals = [rand["cnn_lstm_random"]["acc"] * 100, R["cnn_lstm_loso"]["mean_acc"] * 100]
    labs = ["Random window split\n(same people in train & test)", "Leave-one-subject-out\n(new person)"]
    ax.barh([1, 0], vals, color=[GRID, ORANGE], height=0.55)
    for y_, v in zip([1, 0], vals):
        ax.text(v - 1, y_, f"{v:.1f}%", va="center", ha="right", color=INK, fontweight="bold")
    ax.set_yticks([1, 0], labs); ax.set_xlim(0, 100); ax.set_xlabel("accuracy (%) — same CNN-LSTM, no augmentation")
    fig.suptitle("Same model, two ways to split: the random split flatters it", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(f"{F}/13_leakage.png", dpi=200); plt.close(fig)

# per-class F1 table for the main model and RF
with open(f"{O}/per_class.md", "w") as fh:
    fh.write("| Activity | " + " | ".join(o[1].replace("\n", " ") for o in order) + " |\n|---|" + "---|" * len(order) + "\n")
    for i, a in enumerate(ACTIVITIES):
        fh.write(f"| {a} | " + " | ".join(f"{R[k]['per_class_f1'][i]:.2f}" for k, _, _ in order) + " |\n")
print(open(f"{O}/per_class.md").read())
