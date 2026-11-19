"""Build step (2) prototype: readable triggers for SPD mechanisms of an AlgZoo RNN.
For each alive component, fit a shallow decision tree predicting importance > 0.5 from interpretable per-step features,
score it on held-out inputs (balanced accuracy), and print the rule. argv: target [variant] [model tag] [depth]."""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT / "lab"))
from spd_core import Decomposed, importance
import judge


def load(target, variant, tag):
    meta, W, X = judge.load_target(target, variant, 0)
    N = int(W["seq_len"][0]); H = W["W_h"].shape[0]
    Ws = {k: torch.tensor(W[k].T.copy(), dtype=torch.float64) for k in ("W_x", "W_h", "W_o")}
    d = torch.load(ROOT / f"results/glass/units/u19c_model_{tag}.pt", weights_only=False)
    mods = torch.nn.ModuleDict({k: Decomposed(v, 20).double() for k, v in Ws.items()}); mods.load_state_dict(d["mods"])
    SD = d["SD"]

    def forward(x, mm):
        h = torch.zeros(x.shape[0], H, dtype=x.dtype)
        for t in range(N):
            h = torch.relu(mm("W_x", x[:, t:t + 1]) + mm("W_h", h))
        return mm("W_o", h) / SD
    return meta, W, X, N, H, mods, forward


def step_features(x, W, N, H):
    """Per (sample, step t = 1..N) features for the calls that compute h_t; plus the network's neuron states h_(t-1)."""
    B = len(x); rows = []; names = None
    h = np.zeros((B, H))
    for t in range(1, N + 1):
        prev = x[:, :t - 1]; cur = x[:, t - 1]; upto = x[:, :t]
        mx_p = prev.max(1) if t > 1 else np.full(B, -9.0); s2_p = np.sort(prev, 1)[:, -2] if t > 2 else np.full(B, -9.0)
        rank = (upto > cur[:, None]).sum(1) + 1
        amax = upto.argmax(1); a2 = np.argsort(upto, 1)[:, -2] if t > 1 else np.zeros(B, int)
        f = {"t": np.full(B, t), "x_t": cur, "rank of x_t so far (1 = largest)": rank,
             "x_t - previous max": np.where(t > 1, cur - mx_p, 0), "x_t - previous 2nd": np.where(t > 2, cur - s2_p, 0),
             "steps since the max": t - 1 - amax, "steps since the 2nd": np.where(t > 1, t - 1 - a2, -1)}
        for j in range(H):
            f[f"neuron {j} on before"] = (h[:, j] > 1e-9).astype(float)
        names = list(f); rows.append(np.stack([f[k] for k in names], 1))
        h = np.maximum(cur[:, None] * W["W_x"] + h @ W["W_h"], 0)
    return names, np.concatenate(rows)          # [N*B, F], ordered call-major (call k = step k+1)


def explain(target="zoo_2nd_argmax_8_5", variant="original", tag=None, depth=3, n=16000):
    tag = tag or f"{target}_{variant}_30000_512"
    meta, W, X, N, H, mods, forward = load(target, variant, tag)
    x = np.asarray(X[:n], dtype=np.float64); g = importance(mods, forward, torch.tensor(x))
    names, F = step_features(x, W, N, H)
    half = len(x) // 2; idx = np.arange(N * len(x)); train = (idx % len(x)) < half
    out = {}
    for mname in ("W_h", "W_x"):
        G = g[mname].numpy().reshape(-1, 20)
        for c in range(20):
            if G[:, c].mean() <= 0.01:
                continue
            y = G[:, c] > 0.5
            if y.all() or (~y).all():
                out[f"{mname}:{c}"] = {"rule": "always on" if y.all() else "never on", "balanced_accuracy": 1.0, "on_share": float(y.mean())}
                continue
            res = {}
            for label, cols in (("task features", [i for i, k in enumerate(names) if not k.startswith("neuron")]), ("task + neuron states", list(range(len(names))))):
                tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0).fit(F[train][:, cols], y[train])
                p = tree.predict(F[~train][:, cols]); yt = y[~train]
                ba = 0.5 * ((p[yt]).mean() + (~p[~yt]).mean())
                res[label] = {"balanced_accuracy": float(ba), "rule": export_text(tree, feature_names=[names[i] for i in cols], decimals=2)}
            out[f"{mname}:{c}"] = {"on_share": float(y.mean()), **res}
            print(f"{mname} component {c:2d} (on {100 * y.mean():4.1f}% of calls): held-out balanced accuracy, task features {res['task features']['balanced_accuracy']:.3f}"
                  f" | + neuron states {res['task + neuron states']['balanced_accuracy']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    out = explain(a[0] if a else "zoo_2nd_argmax_8_5", a[1] if len(a) > 1 else "original", a[2] if len(a) > 2 else None, int(a[3]) if len(a) > 3 else 3)
    (ROOT / f"results/glass/units/explain_{a[0] if a else 'zoo_2nd_argmax_8_5'}_{a[1] if len(a) > 1 else 'original'}_d{a[3] if len(a) > 3 else 3}.json").write_text(json.dumps(out, indent=1))
