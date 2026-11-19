"""P5: ε95 (P3's precision requirement) at one intermediate site, across network families (pre-registered in DIARY)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

SITE = {"onelayer_transformer": "resid_mid", "transformer": "resid0.mid", "mlp_relu": "h1", "mlp_tanh": "h1"}

def eps95(target, n=4000):
    meta, W, X = judge.load_target(target, "original", 0); judge.GROUPS = meta.get("out_groups")
    X = np.asarray(X); X = X[np.random.default_rng(0).permutation(len(X))[:n]]; kind = meta["forward"]; site = SITE[kind]
    acts = {}; ref = nets.forward(kind, W, X, acts=acts); sd = np.asarray(acts[site]).std(0)
    for e in np.geomspace(0.001, 1.0, 31):
        agree = np.mean([judge.same(ref, nets.forward(kind, W, X, hook={site: lambda z, r=np.random.default_rng(s): z + r.normal(size=z.shape) * e * sd})).mean() for s in range(3)])
        if agree < 0.95:
            return {"site": site, "coords": int((sd > 0).sum()), "eps95": float(e)}
    return {"site": site, "coords": int((sd > 0).sum()), "eps95": 1.0}

if __name__ == "__main__":
    res = {}
    for t in sys.argv[1:] or ["modadd_p113", "modmul_p113", "tracr_count_greater", "tracr_first_match", "bool_xor", "bool_majority3",
                              "bool_parity3", "bool_compare2", "astra_breakout"]:
        res[t] = eps95(t); print(t, res[t], flush=True)
    (ROOT / "results/glass/units/p5_eps95_cross.json").write_text(json.dumps(res, indent=1))
