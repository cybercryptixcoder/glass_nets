"""U10: train M8,5 2nd-argmax RNNs with and without hidden-state noise; does tolerance make approximate unit accounts
faithful? (pre-registered in DIARY)."""
import sys, json, copy
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab")); sys.path.insert(0, str(ROOT / "lab/targets"))
import judge, nets, unitmap
from build_registry import save
H, N, STEPS, BATCH, SIGMA = 8, 5, int(sys.argv[1]) if len(sys.argv) > 1 else 6000, 1024, 0.1

def train(seed, sigma):
    import torch as th
    th.manual_seed(seed); g = th.Generator().manual_seed(seed)
    Wx = th.nn.Parameter(th.randn(1, H, dtype=th.float64) * 0.5); Wh = th.nn.Parameter(th.randn(H, H, dtype=th.float64) / np.sqrt(H))
    Wo = th.nn.Parameter(th.randn(H, N, dtype=th.float64) / np.sqrt(H))
    opt = th.optim.Adam([Wx, Wh, Wo], lr=3e-3); sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for step in range(STEPS):
        X = th.randn(BATCH, N, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
        s = th.zeros(BATCH, H, dtype=th.float64)
        for t in range(N):
            s = th.relu(X[:, t:t + 1] * Wx + s @ Wh)
            if sigma > 0:
                s = s + sigma * th.randn(s.shape, generator=g, dtype=th.float64) * s.detach().std(0, keepdim=True)
        loss = th.nn.functional.cross_entropy(s @ Wo, y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return {"W_x": Wx.detach().numpy(), "W_h": Wh.detach().numpy(), "W_o": Wo.detach().numpy(), "seq_len": np.array([N], dtype=np.float64)}

def save_target(tid, W):
    X = np.random.default_rng(0).standard_normal((8000, N)); labels = np.argsort(X, -1)[:, -2]
    acc = float((nets.forward("rnn_relu", W, X).argmax(1) == labels).mean())
    meta = copy.deepcopy(json.loads((ROOT / "lab/models/zoo_2nd_argmax_8_5/meta.json").read_text()))
    for k in ("id", "tensors", "params"): meta.pop(k, None)
    meta.update({"name": f"U10 M8,5 ({tid})", "source": "trained by scripts/glass/units/u10_tolerance.py", "test_acc": acc,
                 "answer": {"short": meta["answer"]["short"], "detail": f"U10 experiment network {tid}; accuracy {acc:.4f}."}})
    d = ROOT / "lab/models" / tid; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "inputs.npy", X.astype(np.float32)); np.save(d / "labels.npy", labels.astype(np.int64)); save(tid, meta, W)
    return acc

def joint(tid, m):
    meta, W, X = judge.load_target(tid); X = X[:8000]; Xt = X[4000:]; ref = unitmap.run(W, Xt); rep = {}
    later = [c for c in m["cells"] if c["t"] > 1]; tot = sum(c["var"] for c in later) or 1.0; used = 0.0
    for c in later:
        if c["fit"] < 0.9 or not c["prims"]: continue
        t, p, w = c["t"], c["prims"], np.array(c["w"])
        rep[(t, c["unit"])] = (lambda X_, t=t, w=w: np.maximum(X_[:, :t] @ w, 0)) if p[0].startswith("x_") and p[0][2:].isdigit() \
            else (lambda X_, t=t, p=p, w=w: np.maximum(np.stack([unitmap.library(X_, t)[q] for q in p], 1) @ w, 0))
        used += c["var"]
    return float(judge.same(ref, unitmap.run(W, Xt, rep)).mean()), used / tot, len(rep)

if __name__ == "__main__":
    # U10: argv = [steps]; U10b: argv = [steps, comma-separated sigmas, id prefix]
    out = {}
    levels = [("std", 0.0), ("noise", SIGMA)] if len(sys.argv) < 3 else [(f"sig{v}", float(v)) for v in sys.argv[2].split(",")]
    prefix = "u10_m85" if len(sys.argv) < 4 else sys.argv[3]
    for seed in range(3):
        for kind, sigma in levels:
            tid = f"{prefix}_{kind}_s{seed}"; acc = save_target(tid, train(seed, sigma))
            m = unitmap.unit_map(tid); js, share, k = joint(tid, m)
            out[tid] = {"acc": acc, "eps95": m["eps95"], "coverage_history": m["coverage_history"], "coverage_raw": m["coverage_raw"],
                        "joint_substitution": js, "substituted_var_share": share, "n_substituted": k, "per_unit": m["per_unit"]}
            print(f"{tid}: acc {acc:.3f} eps95 {m['eps95']:.4f} cov hist {m['coverage_history']:.2f} raw {m['coverage_raw']:.2f} | joint substitution {100*js:.1f}% "
                  f"({k} neuron-steps, {100*share:.0f}% of variance)", flush=True)
    (ROOT / f"results/glass/units/{prefix}_{STEPS}{'_' + sys.argv[2].replace(',', '-') if len(sys.argv) > 2 else ''}.json").write_text(json.dumps(out, indent=1))
