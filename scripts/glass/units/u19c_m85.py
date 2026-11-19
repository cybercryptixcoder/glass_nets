"""U19(c): SPD on an AlgZoo RNN (pre-registered in docs/glass/DIARY.md). argv: target [variant] [steps].
Decomposes W_x, W_h, W_o; reports alive components, sparsity of use, CI-masked decision fidelity, and whether each alive
W_h component's importance is predicted by a simple task event (balanced accuracy >= 0.9)."""
import sys, json
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT / "lab"))
from spd_core import decompose, importance, _run
import judge

target = sys.argv[1] if len(sys.argv) > 1 else "zoo_2nd_argmax_8_5"
variant = sys.argv[2] if len(sys.argv) > 2 else "original"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
meta, W, X = judge.load_target(target, variant, 0)
N = int(W["seq_len"][0]); H = W["W_h"].shape[0]
Wx, Wh, Wo = (torch.tensor(W[k].T.copy(), dtype=torch.float64) for k in ("W_x", "W_h", "W_o"))   # as [d_out, d_in]
with torch.no_grad():
    Xs = torch.tensor(np.asarray(X[:8000]), dtype=torch.float64)
    h = torch.zeros(len(Xs), H, dtype=torch.float64)
    for t in range(N):
        h = torch.relu(Xs[:, t:t + 1] @ Wx.T + h @ Wh.T)
    SD = float((h @ Wo.T).std())


def forward(x, mm):
    h = torch.zeros(x.shape[0], H, dtype=x.dtype)
    for t in range(N):
        h = torch.relu(mm("W_x", x[:, t:t + 1]) + mm("W_h", h))
    return mm("W_o", h) / SD


def sample(n):
    return torch.randn(n, N, dtype=torch.float64)


torch.manual_seed(0)
mods = decompose(forward, {"W_x": Wx, "W_h": Wh, "W_o": Wo}, sample, C=20, steps=steps, coef_min=3e-3, p=1.0,
                 coef_faith=1.0, lr=1e-3, batch=batch, seed=0)
tag = f"{target}_{variant}_{steps}_{batch}"
torch.save({"mods": mods.state_dict(), "SD": SD}, ROOT / f"results/glass/units/u19c_model_{tag}.pt")

xs = torch.tensor(np.asarray(X[8000:16000] if len(X) >= 16000 else X[:8000]), dtype=torch.float64)
g = importance(mods, forward, xs)                                   # {name: [calls, B, C]}
with torch.no_grad():
    ref = _run(forward, xs, mods, lambda n, k: None).numpy()
    masked = _run(forward, xs, mods, lambda n, k: g[n][k]).numpy()
    alive = {n: [c for c in range(20) if g[n][:, :, c].mean() > 0.01] for n in mods}
    am = {n: torch.zeros(20, dtype=torch.float64).index_fill_(0, torch.tensor(alive[n], dtype=torch.long), 1.0) if alive[n] else torch.zeros(20, dtype=torch.float64) for n in mods}
    alive_only = _run(forward, xs, mods, lambda n, k: am[n].expand(len(xs), 20)).numpy()
fid_ci = float(judge.same(ref, masked).mean()); fid_alive = float(judge.same(ref, alive_only).mean())
l0 = {n: float((g[n] > 0.1).double().sum(-1).mean()) for n in mods}

# events at each W_h call k (step t = k + 1, the call that turns h_(t-1) into h_t): computed from the inputs
x = xs.numpy(); events = {}
for k in range(N):
    t = k + 1; prev = x[:, :t - 1]; cur = x[:, t - 1]
    mx = prev.max(1) if t > 1 else np.full(len(x), -np.inf); s2 = np.sort(prev, 1)[:, -2] if t > 2 else np.full(len(x), -np.inf)
    am_prev = prev.argmax(1) if t > 1 else np.full(len(x), -1)
    ev = {"x_t is a new max": cur > mx, "x_t is a new 2nd": (cur < mx) & (cur > s2), "x_t below the 2nd": cur < s2,
          "max was at t-1": am_prev == t - 2, "x_t > 0": cur > 0}
    for e, v in ev.items():
        events.setdefault(e, []).append(v)
events = {e: np.concatenate(v) for e, v in events.items()}
labels = {}
G = g["W_h"].numpy().reshape(-1, 20)
for c in alive["W_h"]:
    on = G[:, c] > 0.5; best = ("none", 0.0)
    for e, v in events.items():
        for name, vv in ((e, v), ("not " + e, ~v)):
            if on.any() and (~on).any():
                ba = 0.5 * (vv[on].mean() + (~vv[~on]).mean())
                if ba > best[1]:
                    best = (name, float(ba))
    labels[c] = best
res = {"target": target, "variant": variant, "steps": steps, "alive": {n: len(v) for n, v in alive.items()}, "l0_per_call": l0,
       "fidelity_ci_masked": fid_ci, "fidelity_alive_only": fid_alive, "task_rule": float(meta["test_acc"]),
       "wh_labels": {str(c): v for c, v in labels.items()},
       "wh_labelled_share": (sum(1 for v in labels.values() if v[1] >= 0.9) / len(labels)) if labels else 0.0}
print(json.dumps(res, indent=1))
(ROOT / f"results/glass/units/u19c_{tag}.json").write_text(json.dumps(res, indent=1))
