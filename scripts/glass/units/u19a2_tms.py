"""U19(a'): SPD known-answer test on TMS 5-2 with the revised criterion (pre-registered in DIARY); argv[8] = seed."""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from spd_core import decompose, importance

ROOT = Path(__file__).resolve().parents[3]
K, D, P_ON = 5, 2, 0.05
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 1
torch.manual_seed(SEED)


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
mods = decompose(forward, {"W1": W1, "W2": W2}, sample, seed=SEED, **cfg)
torch.save({"mods": mods.state_dict(), "W1": W1, "bb": bb, "cfg": cfg}, ROOT / f"results/glass/units/u19a2_model_seed{SEED}.pt")

# 2. revised known-answer check (U19(a') in DIARY)
E = torch.eye(K, dtype=torch.float64)
xs = sample(20000); g = {k: v[0] for k, v in importance(mods, forward, xs).items()}; on = xs > 0
with torch.no_grad():
    yt = forward(xs, lambda name, a: mods[name].apply(a))
needed = on & (yt > 0.05)
res = {"config": cfg, "seed": SEED, "matrices": {}}; ok_all = True
for name, m in mods.items():
    alive = [c for c in range(cfg["C"]) if g[name][:, c].mean() > 0.01]
    with torch.no_grad():
        S = sum(torch.outer(m.U[c], m.V[c]) for c in alive) if alive else torch.zeros_like(m.W)
        rel = float((m.W - S).norm() / m.W.norm())
        dirs = {c: (m.V[c] if name == "W1" else m.U[c]) for c in alive}
        cos = {c: (E @ d / d.norm()).abs() for c, d in dirs.items()}
    match = {c: (int(v.argmax()), float(v.max())) for c, v in cos.items()}
    bij = len(alive) == K and sorted(v[0] for v in match.values()) == list(range(K))
    dir_ok = all(v[1] >= 0.9 for v in match.values()); rec_ok = rel <= 0.1
    ci_rows = []; ci_ok = True
    for c, (j, _) in match.items():
        only_other = (on.sum(1) == 1) & (~on[:, j])
        hi = float((g[name][needed[:, j], c] > 0.5).double().mean()); lo = float((g[name][only_other, c] < 0.1).double().mean())
        ci_rows.append((c, j, round(hi, 3), round(lo, 3))); ci_ok &= hi >= 0.9 and lo >= 0.9
    passed = bij and dir_ok and rec_ok and ci_ok; ok_all &= passed
    res["matrices"][name] = {"alive": len(alive), "alive_recon_rel_error": rel, "match": {str(c): v for c, v in match.items()}, "ci": ci_rows, "pass": passed}
    print(f"{name}: alive {len(alive)} | alive-only reconstruction error {rel:.3f} | direction matches {[(c, v[0], round(v[1], 3)) for c, v in match.items()]}")
    print(f"   CI (component, feature, share high when needed, share low when only another is on): {ci_rows}")
    print(f"   one-to-one {bij}, directions {dir_ok}, reconstruction {rec_ok}, CI {ci_ok} -> {'PASS' if passed else 'FAIL'}", flush=True)
res["pass"] = ok_all; print("KNOWN ANSWER (revised)", "PASS" if ok_all else "FAIL")
(ROOT / f"results/glass/units/u19a2_tms_seed{SEED}.json").write_text(json.dumps(res, indent=1))
