"""Training + evaluation.

Protocols
- loso   : leave-one-subject-out, 10 folds. Inside each fold one of the 9
           training subjects is held out as validation for early stopping.
           This answers "does it work on a person it has never seen?"
- random : random 80/20 split of windows (stratified). Included ONLY to show
           how much overlap/subject leakage inflates the score.
"""
import argparse, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from models import MODELS, n_params

torch.set_num_threads(2)
D = np.load(str(ROOT) + "/windows.npz")
X, Y, G = D["X"], D["y"], D["g"]
NC = 12


MAG_GROUPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]


def add_mag(x):
    """Append |v| of each 3-axis sensor -> 20 channels. Magnitude does not change
    when the sensor is rotated, so it is robust to how the strap was worn."""
    mags = [np.linalg.norm(x[..., list(g)], axis=-1, keepdims=True) for g in MAG_GROUPS]
    return np.concatenate([x] + mags, axis=-1).astype(np.float32)


def standardise(tr, *others):
    mu = tr.reshape(-1, tr.shape[-1]).mean(0)
    sd = tr.reshape(-1, tr.shape[-1]).std(0) + 1e-6
    return [(a - mu) / sd for a in (tr, *others)]   # stats from TRAIN only


def predict(model, x, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i+bs])).argmax(1).numpy())
    return np.concatenate(out)


def rand_rot(rng, max_deg):
    """Random 3-D rotation matrix: random axis, angle in [-max_deg, max_deg]."""
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K).astype(np.float32)


# sensor groups: (acc cols, gyro cols or None). Same rotation for acc+gyro of one sensor.
SENSORS = [((0, 1, 2), None), ((3, 4, 5), (6, 7, 8)), ((9, 10, 11), (12, 13, 14))]


def augment(xb, rng, mu, sd, max_deg=20, mag=False):
    """xb is RAW (unnormalised). Simulates a strap worn slightly rotated,
    a different body size/strength (scaling) and sensor noise (jitter)."""
    xb = xb.copy()
    for i in range(len(xb)):
        for acc, gyr in SENSORS:
            R = rand_rot(rng, max_deg)
            xb[i][:, acc] = xb[i][:, acc] @ R.T
            if gyr: xb[i][:, gyr] = xb[i][:, gyr] @ R.T
    xb *= rng.uniform(0.9, 1.1, size=(len(xb), 1, xb.shape[2])).astype(np.float32)
    if mag: xb = add_mag(xb)
    xb = (xb - mu) / sd
    xb += rng.normal(0, 0.05, size=xb.shape).astype(np.float32)
    return xb


def fit(name, xtr, ytr, xva, yva, epochs=40, patience=8, seed=0, log=None, raw_tr=None, musd=None, mag=False):
    """If raw_tr/musd are given, each batch is augmented on the fly from raw data."""
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = MODELS[name](c_in=xtr.shape[-1])
    w = len(ytr) / (NC * np.bincount(ytr, minlength=NC).clip(1))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr).long()
    best, best_state, bad = -1, None, 0
    hist = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt)); tl = 0
        for i in range(0, len(xt), 64):
            idx = perm[i:i+64]
            xb = xt[idx] if raw_tr is None else torch.from_numpy(augment(raw_tr[idx.numpy()], rng, *musd, mag=mag))
            opt.zero_grad()
            loss = lossf(model(xb), yt[idx])
            loss.backward(); opt.step(); tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(torch.from_numpy(xva)), torch.from_numpy(yva).long()).item()
        vf1 = f1_score(yva, predict(model, xva), average="macro")
        hist.append(dict(epoch=ep+1, train_loss=tl/len(xt), val_loss=vl, val_f1=vf1))
        if vf1 > best:
            best, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def run_loso(name, epochs, aug=False, mag=False):
    tag = ("_aug" if aug else "") + ("_mag" if mag else "")
    subj = np.unique(G); res = []; yt_all, yp_all = [], []; hists = {}
    for s in subj:
        t0 = time.time()
        val_s = subj[(np.where(subj == s)[0][0] + 1) % len(subj)]  # next subject
        tr = (G != s) & (G != val_s); va = G == val_s; te = G == s
        f = add_mag if mag else (lambda a: a)
        xtr, xva, xte = standardise(f(X[tr]), f(X[va]), f(X[te]))
        kw = dict(mag=mag)
        if aug:
            flat = f(X[tr]).reshape(-1, xtr.shape[-1])
            kw.update(raw_tr=X[tr], musd=(flat.mean(0), flat.std(0) + 1e-6))
        model, h = fit(name, xtr, Y[tr], xva, Y[va], epochs=epochs, patience=12 if aug else 8, **kw)
        yp = predict(model, xte)
        r = dict(test_subject=int(s), acc=accuracy_score(Y[te], yp),
                 f1=f1_score(Y[te], yp, average="macro"), epochs=len(h),
                 secs=round(time.time()-t0, 1))
        print(name, r, flush=True)
        res.append(r); yt_all.append(Y[te]); yp_all.append(yp); hists[int(s)] = h
        if s == subj[0]:
            torch.save(model.state_dict(), f"{ROOT}/out/{name}{tag}_fold1.pt")
    yt_all, yp_all = np.concatenate(yt_all), np.concatenate(yp_all)
    return dict(model=name + tag, params=n_params(MODELS[name](c_in=20 if mag else 15)), protocol="loso",
                folds=res,
                mean_acc=float(np.mean([r["acc"] for r in res])),
                std_acc=float(np.std([r["acc"] for r in res])),
                mean_f1=float(np.mean([r["f1"] for r in res])),
                std_f1=float(np.std([r["f1"] for r in res])),
                pooled_acc=accuracy_score(yt_all, yp_all),
                pooled_f1=f1_score(yt_all, yp_all, average="macro"),
                per_class_f1=f1_score(yt_all, yp_all, average=None).tolist(),
                cm=confusion_matrix(yt_all, yp_all).tolist(), hist=hists)


def run_random(name, epochs):
    idx = np.arange(len(Y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=Y, random_state=0)
    tr, va = train_test_split(tr, test_size=0.125, stratify=Y[tr], random_state=0)
    xtr, xva, xte = standardise(X[tr], X[va], X[te])
    model, h = fit(name, xtr, Y[tr], xva, Y[va], epochs=epochs)
    yp = predict(model, xte)
    r = dict(model=name, protocol="random", acc=accuracy_score(Y[te], yp),
             f1=f1_score(Y[te], yp, average="macro"))
    print(r, flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_lstm")
    ap.add_argument("--protocol", default="loso")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--mag", action="store_true")
    a = ap.parse_args()
    if a.protocol == "loso":
        out = run_loso(a.model, a.epochs, a.aug, a.mag)
    else:
        out = run_random(a.model, a.epochs)
    tag = ("_aug" if a.aug else "") + ("_mag" if a.mag else "")
    json.dump(out, open(f"{ROOT}/out/{a.model}{tag}_{a.protocol}.json", "w"), indent=1)
