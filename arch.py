"""CNN-LSTM architecture diagram with tensor shapes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
blocks = [
    ("Input window", "100 × 20\n15 acc/gyro + 5 |v|", "#cde2fb", "#2a78d6"),
    ("Conv1D 64, k=5\n+ BN + ReLU", "100 × 64", "#d4f0e4", "#1baf7a"),
    ("Conv1D 64, k=5\n+ BN + ReLU", "100 × 64", "#d4f0e4", "#1baf7a"),
    ("MaxPool 2\n+ Dropout 0.5", "50 × 64", "#d4f0e4", "#1baf7a"),
    ("LSTM 64", "last hidden\nstate: 64", "#e1dcf7", "#6250d6"),
    ("Dropout 0.5\n+ Dense 12", "12 logits\n→ softmax", "#f9d6d6", "#e34948"),
]
fig, ax = plt.subplots(figsize=(14, 3.6)); fig.patch.set_facecolor(SURF)
ax.set_xlim(0, 14); ax.set_ylim(0, 3.6); ax.axis("off")
W, H, gap = 1.95, 1.7, 0.36
for i, (t, s, fc, ec) in enumerate(blocks):
    x = 0.2 + i * (W + gap)
    ax.add_patch(FancyBboxPatch((x, 1.2), W, H, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.8))
    ax.text(x + W / 2, 1.2 + H * 0.64, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + W / 2, 1.2 + H * 0.2, s, ha="center", va="center", fontsize=9, color=INK2)
    if i < len(blocks) - 1:
        ax.annotate("", (x + W + gap - 0.02, 2.05), (x + W + 0.02, 2.05), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
ax.text(0.2 + 1.5 * (W + gap) + W / 2, 0.75, "CNN: learns local motion shapes (≈0.1 s kernels)", ha="center", fontsize=10, color="#1baf7a", fontweight="bold")
ax.text(0.2 + 4 * (W + gap) + W / 2, 0.75, "LSTM: how they evolve over 2 s", ha="center", fontsize=10, color="#6250d6", fontweight="bold")
ax.text(0.2, 3.3, "CNN-LSTM, 61,324 trainable parameters  (shapes = time steps × channels)", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(Path(__file__).parent / "fig/14_architecture.png", dpi=200)
