"""MHEALTH loading, channel selection and windowing.

Design choices (each one is a viva question):
- Channels: only accelerometer + gyroscope (15 ch), as the problem statement asks.
  ECG (cols 4-5) and magnetometer are dropped. Chest sensor has no gyro.
- Label 0 = "null" (transition / not part of protocol) -> dropped.
- Windows: 2.0 s = 100 samples @ 50 Hz, 50% overlap (stride 50).
  Windows are cut only inside a contiguous run of one label, so every window
  has exactly one true label (no mixed windows).
- Split is BY SUBJECT. Windows from the same person never appear in both
  train and test, otherwise overlapping windows leak and accuracy is inflated.
"""
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

DATA_DIR = os.environ.get("MHEALTH_DIR", str(ROOT / "MHEALTHDATASET"))
FS = 50

# 0-indexed columns in the .log file
CHANNELS = {
    "chest_acc_x": 0, "chest_acc_y": 1, "chest_acc_z": 2,
    "ankle_acc_x": 5, "ankle_acc_y": 6, "ankle_acc_z": 7,
    "ankle_gyro_x": 8, "ankle_gyro_y": 9, "ankle_gyro_z": 10,
    "arm_acc_x": 14, "arm_acc_y": 15, "arm_acc_z": 16,
    "arm_gyro_x": 17, "arm_gyro_y": 18, "arm_gyro_z": 19,
}
CH_NAMES = list(CHANNELS)
CH_IDX = list(CHANNELS.values())
LABEL_COL = 23

ACTIVITIES = [
    "Standing", "Sitting", "Lying", "Walking", "Climbing stairs",
    "Waist bend", "Arm elevation", "Knee bend", "Cycling",
    "Jogging", "Running", "Jump front/back",
]  # labels 1..12 -> class 0..11


def load_subject(s):
    a = np.loadtxt(f"{DATA_DIR}/mHealth_subject{s}.log")
    return a[:, CH_IDX].astype(np.float32), a[:, LABEL_COL].astype(int)


def runs(y):
    """Yield (start, end, label) for contiguous runs of the same label."""
    change = np.flatnonzero(np.diff(y)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(y)]
    for s, e in zip(starts, ends):
        yield s, e, y[s]


def window_subject(X, y, win=100, stride=50):
    Xs, ys = [], []
    for s, e, lab in runs(y):
        if lab == 0:
            continue
        for t in range(s, e - win + 1, stride):
            Xs.append(X[t:t + win])
            ys.append(lab - 1)
    return np.stack(Xs), np.array(ys)


def build(subjects=range(1, 11), win=100, stride=50):
    """Returns X (N, win, C), y (N,), g (N,) subject id."""
    Xa, ya, ga = [], [], []
    for s in subjects:
        X, y = load_subject(s)
        Xw, yw = window_subject(X, y, win, stride)
        Xa.append(Xw); ya.append(yw); ga.append(np.full(len(yw), s))
    return np.concatenate(Xa), np.concatenate(ya), np.concatenate(ga)


if __name__ == "__main__":
    X, y, g = build()
    np.savez_compressed(str(ROOT) + "/windows.npz", X=X, y=y, g=g)
    print(X.shape, np.bincount(y), np.bincount(g))
