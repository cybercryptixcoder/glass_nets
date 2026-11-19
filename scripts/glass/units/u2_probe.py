"""U2: linear decodability of task quantities from the RNN hidden state (distributed units)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def quantities(X, t):
    H = X[:, :t]; o = np.argsort(-H, 1); s = np.take_along_axis(H, o, 1)
    q = {"max": s[:, 0], "2nd": s[:, 1] if t > 1 else s[:, 0], "min": s[:, -1], "x_t": H[:, -1],
         "age_max": (t - 1 - o[:, 0]).astype(float), "age_2nd": (t - 1 - (o[:, 1] if t > 1 else o[:, 0])).astype(float),
         "median": np.median(H, 1), "age_median": (t - 1 - np.argsort(H, 1)[:, (t - 1) // 2]).astype(float)}
    return q

def r2(Ftr, ytr, Fte, yte):
    A = np.c_[Ftr, np.ones(len(Ftr))]; w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    p = np.c_[Fte, np.ones(len(Fte))] @ w; v = yte.var(); return 1 - ((yte - p) ** 2).mean() / v if v > 0 else float("nan")

out = {}
for target in sys.argv[1:] or ["zoo_2nd_argmax_4_3", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_16_10", "zoo_argmedian_8_5", "zoo_argmedian_16_5"]:
    for var in ("original", "shuffled"):
        meta, W, X = judge.load_target(target, var, 0); X = X[:8000]; T = X.shape[1]
        acts = {}; nets.forward("rnn_relu", W, X, acts=acts)
        H = np.asarray(acts[f"h{T}"]).reshape(len(X), -1); Q = quantities(X, T)
        out[f"{target}|{var}"] = {k: round(float(r2(H[:4000], v[:4000], H[4000:], v[4000:])), 3) for k, v in Q.items()}
        print(f"{target:22s} {var:9s} last step: " + "  ".join(f"{k} {v:.2f}" for k, v in out[f"{target}|{var}"].items()), flush=True)
(ROOT / "results/glass/units/u2.json").write_text(json.dumps(out, indent=1))
