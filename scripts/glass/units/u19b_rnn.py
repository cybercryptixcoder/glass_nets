"""U19(b): SPD known-answer test on a ReLU RNN with planted mechanisms (pre-registered in docs/glass/DIARY.md).

The planted RNN accumulates k sparse features over n steps: h_t = relu(W_x x_t + W_h h_(t-1)), out = W_o h_n, with
W_x = W_h = W_o = I (k x k). Inputs: at each step each feature is on with probability P_ON (value U[0,1]).
Planted mechanisms, per matrix: e_j e_j^T (one per feature j). A mechanism is needed exactly when its input
coordinate is non-zero: W_x's at the steps where feature j arrives, W_h's at steps after feature j has arrived,
W_o's when feature j arrived at all.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from spd_core import decompose, importance

ROOT = Path(__file__).resolve().parents[3]
K, N, P_ON = 5, 6, 0.03
torch.manual_seed(0)
I = torch.eye(K, dtype=torch.float64)


def sample(n):
    on = (torch.rand(n, N, K, dtype=torch.float64) < P_ON).double()
    return on * torch.rand(n, N, K, dtype=torch.float64)


def forward(x, mm):
    h = torch.zeros(x.shape[0], K, dtype=x.dtype)
    for t in range(N):
        h = torch.relu(mm("W_x", x[:, t]) + mm("W_h", h))
    return mm("W_o", h)


args = sys.argv[1:]
cfg = dict(C=int(args[0]) if len(args) > 0 else 20, steps=int(args[1]) if len(args) > 1 else 30000,
           coef_min=float(args[2]) if len(args) > 2 else 3e-3, p=float(args[3]) if len(args) > 3 else 1.0,
           coef_faith=float(args[4]) if len(args) > 4 else 1.0, lr=float(args[5]) if len(args) > 5 else 1e-3,
           batch=int(args[6]) if len(args) > 6 else 2048)
mods = decompose(forward, {"W_x": I, "W_h": I, "W_o": I}, sample, **cfg)
torch.save({"mods": mods.state_dict(), "cfg": cfg}, ROOT / f"results/glass/units/u19b_model_{'_'.join(args) or 'default'}.pt")

# known-answer check: per matrix, the inputs of each call; a mechanism j is "needed" where input coordinate j > 0
xs = sample(20000); g = importance(mods, forward, xs)       # {name: [calls, B, C]}
inputs = {"W_x": [xs[:, t] for t in range(N)]}
h = torch.zeros(len(xs), K, dtype=torch.float64); hs = []
for t in range(N):
    hs.append(h.clone()); h = torch.relu(xs[:, t] + h)
inputs["W_h"] = hs; inputs["W_o"] = [h]
res = {"config": cfg, "matrices": {}}; ok_all = True
for name, m in mods.items():
    G = g[name].reshape(-1, cfg["C"]); Ain = torch.cat(inputs[name])               # [calls*B, C], [calls*B, K]
    A = Ain > 0
    NEED = Ain >= 0.2      # U19(b') criterion: a mechanism is needed when its input is >= 0.2 (SPD importance is causal,
                           # so tiny contributions may legitimately get low importance; see DIARY U19(a'))
    alive = [c for c in range(cfg["C"]) if G[:, c].mean() > 0.01]
    truth = [torch.outer(I[j], I[j]) for j in range(K)]
    cos = np.array([[float((torch.outer(m.U[c], m.V[c]).detach() * t).sum() / (torch.outer(m.U[c], m.V[c]).detach().norm() * t.norm() + 1e-12)) for t in truth] for c in alive])
    match = {c: (int(np.argmax(np.abs(cos[i]))), float(cos[i].max())) for i, c in enumerate(alive)}
    bij = len(alive) == K and sorted(v[0] for v in match.values()) == list(range(K))
    cos_ok = all(abs(v[1]) >= 0.9 for v in match.values())
    ci_rows = []; ci_ok = True
    for c, (j, _) in match.items():
        other_only = NEED.any(1) & (~A[:, j])
        hi = float((G[NEED[:, j], c] > 0.5).double().mean()) if NEED[:, j].any() else 0.0
        lo = float((G[other_only, c] < 0.1).double().mean()) if other_only.any() else 1.0
        ci_rows.append((c, j, round(hi, 3), round(lo, 3))); ci_ok &= hi >= 0.9 and lo >= 0.9
    passed = bij and cos_ok and ci_ok; ok_all &= passed
    res["matrices"][name] = {"alive": len(alive), "match": {str(c): v for c, v in match.items()}, "ci": ci_rows, "pass": passed}
    print(f"{name}: alive {len(alive)} | matches {[(c, v[0], round(v[1], 3)) for c, v in match.items()]}")
    print(f"   CI (component, mechanism, share high when needed, share low when only others are needed): {ci_rows}")
    print(f"   one-to-one {bij}, cosine >= 0.9 {cos_ok}, CI {ci_ok} -> {'PASS' if passed else 'FAIL'}", flush=True)
res["pass"] = ok_all; print("KNOWN ANSWER", "PASS" if ok_all else "FAIL")
(ROOT / f"results/glass/units/u19b_rnn_{'_'.join(args) or 'default'}.json").write_text(json.dumps(res, indent=1))
