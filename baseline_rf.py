"""Classical baseline: hand-crafted statistical features + Random Forest,
same leave-one-subject-out protocol. If the deep model can't beat this,
the deep model isn't justified."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

D = np.load(str(ROOT) + "/windows.npz")
X, Y, G = D["X"], D["y"], D["g"]


def feats(x):  # x: (N, T, C)
    f = [x.mean(1), x.std(1), x.min(1), x.max(1),
         np.percentile(x, 25, axis=1), np.percentile(x, 75, axis=1),
         np.abs(np.diff(x, axis=1)).mean(1)]
    # magnitude of each 3-axis sensor (orientation-independent)
    for k in range(0, 15, 3):
        m = np.linalg.norm(x[:, :, k:k+3], axis=2)
        f += [m.mean(1, keepdims=True), m.std(1, keepdims=True)]
    return np.concatenate(f, axis=1)


F = feats(X)
res, yt, yp = [], [], []
for s in np.unique(G):
    tr, te = G != s, G == s
    rf = RandomForestClassifier(300, n_jobs=2, random_state=0, class_weight="balanced")
    rf.fit(F[tr], Y[tr]); p = rf.predict(F[te])
    res.append(dict(test_subject=int(s), acc=accuracy_score(Y[te], p),
                    f1=f1_score(Y[te], p, average="macro")))
    print(res[-1], flush=True); yt.append(Y[te]); yp.append(p)
yt, yp = np.concatenate(yt), np.concatenate(yp)
out = dict(model="rf", protocol="loso", params=None, folds=res, n_features=F.shape[1],
           mean_acc=float(np.mean([r["acc"] for r in res])), std_acc=float(np.std([r["acc"] for r in res])),
           mean_f1=float(np.mean([r["f1"] for r in res])), std_f1=float(np.std([r["f1"] for r in res])),
           pooled_acc=accuracy_score(yt, yp), pooled_f1=f1_score(yt, yp, average="macro"),
           per_class_f1=f1_score(yt, yp, average=None).tolist(),
           cm=confusion_matrix(yt, yp).tolist())
json.dump(out, open(str(ROOT) + "/out/rf_loso.json", "w"), indent=1)
print(out["mean_acc"], out["mean_f1"])
