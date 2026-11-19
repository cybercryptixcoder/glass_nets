"""U19(a): SPD known-answer test on a toy model of superposition (TMS 5-2). Pre-registered in docs/glass/DIARY.md."""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from spd_core import decompose, importance

ROOT = Path(__file__).resolve().parents[3]
K, D, P_ON = 5, 2, 0.05
torch.manual_seed(0)


def sample(n):
    on = (torch.rand(n, K, dtype=torch.float64) < P_ON).double()
    return on * torch.rand(n, K, dtype=torch.float64)


# 1. train the target TMS model (tied weights, as in Elhage et al. 2022)
W = torch.nn.Parameter(torch.randn(D, K, dtype=torch.float64) * 0.5); b = torch.nn.Parameter(torch.zeros(K, dtype=torch.float64))
opt = torch.optim.Adam([W, b], lr=1e-2)
for s in range(8000):
    x = sample(1024); y = torch.relu(x @ W.T @ W + b); loss = ((y - x) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
W1, W2, bb = W.detach().clone(), W.detach().T.clone(), b.detach().clone()
print("TMS trained: loss", round(loss.item(), 5), "column norms", W1.norm(dim=0).numpy().round(2), flush=True)


def forward(x, mm):
    return torch.relu(mm("W2", mm("W1", x)) + bb)


cfg = dict(C=int(sys.argv[1]) if len(sys.argv) > 1 else 20, steps=int(sys.argv[2]) if len(sys.argv) > 2 else 20000,
           coef_min=float(sys.argv[3]) if len(sys.argv) > 3 else 3e-3, p=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
           coef_faith=float(sys.argv[5]) if len(sys.argv) > 5 else 1e3, lr=float(sys.argv[6]) if len(sys.argv) > 6 else 2e-3,
           batch=int(sys.argv[7]) if len(sys.argv) > 7 else 2048)
mods = decompose(forward, {"W1": W1, "W2": W2}, sample, **cfg)

# 2. the known-answer check
truth = {"W1": [torch.outer(W1[:, j], torch.eye(K, dtype=torch.float64)[j]) for j in range(K)],
         "W2": [torch.outer(torch.eye(K, dtype=torch.float64)[j], W1[:, j]) for j in range(K)]}
xs = sample(20000); g = {k: v[0] for k, v in importance(mods, forward, xs).items()}; on = xs > 0
res = {"config": cfg, "matrices": {}}; ok_all = True
for name, m in mods.items():
    alive = [c for c in range(cfg["C"]) if g[name][:, c].mean() > 0.01]
    comps = {c: torch.outer(m.U[c], m.V[c]).detach() for c in alive}
    cos = np.array([[float((comps[c] * t).sum() / (comps[c].norm() * t.norm() + 1e-12)) for t in truth[name]] for c in alive])
    match = {}
    for i, c in enumerate(alive):
        j = int(np.argmax(np.abs(cos[i]))); match[c] = (j, float(cos[i, j]))
    js = [v[0] for v in match.values()]
    bij = len(alive) == K and sorted(js) == list(range(K))
    cos_ok = all(abs(v[1]) >= 0.9 for v in match.values())
    ci_ok = True; ci_rows = []
    for c, (j, _) in match.items():
        only_other = (on.sum(1) == 1) & (~on[:, j])
        hi = float((g[name][on[:, j], c] > 0.5).double().mean()); lo = float((g[name][only_other, c] < 0.1).double().mean())
        ci_rows.append((c, j, round(hi, 3), round(lo, 3))); ci_ok &= hi >= 0.9 and lo >= 0.9
    passed = bij and cos_ok and ci_ok; ok_all &= passed
    res["matrices"][name] = {"alive": len(alive), "match": {str(c): v for c, v in match.items()}, "ci": ci_rows, "pass": passed}
    print(f"{name}: alive {len(alive)} | matches (component -> feature, cosine): {[(c, v[0], round(v[1], 3)) for c, v in match.items()]}")
    print(f"   CI (component, feature, share high when on, share low when only another is on): {ci_rows}")
    print(f"   one-to-one {bij}, cosine >= 0.9 {cos_ok}, CI {ci_ok} -> {'PASS' if passed else 'FAIL'}", flush=True)
res["pass"] = ok_all; print("KNOWN ANSWER", "PASS" if ok_all else "FAIL")
(ROOT / f"results/glass/units/u19a_tms_{'_'.join(sys.argv[1:]) or 'default'}.json").write_text(json.dumps(res, indent=1))
