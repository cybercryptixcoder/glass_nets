"""Exploration for (iii), the trusted-trigger share, on the A3 development networks (design only; claims need held-out A18).
Variants: tree depth; confidence measured on crisp labels only (importance > 0.9 vs < 0.1, the ambiguous middle excluded);
trees trained on crisp labels only."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
from sklearn.tree import DecisionTreeClassifier

def bal(p, y):
    return 0.5 * (p[y].mean() + (~p[~y]).mean()) if y.any() and (~y).any() else float((p == y).mean())

def sweep(tid, spdp):
    tgt = atlas.make_target(tid); mods = atlas.load_decomposition(tgt, spdp)
    X = tgt.X[:16000]; half = len(X) // 2
    tr, te = atlas._network_calls(tgt, mods, X[:half]), atlas._network_calls(tgt, mods, X[half:])
    res = {}
    for k in ("W_x", "W_h", "W_o"):
        _, Ftr, Gtr = tr[k]; _, Fte, Gte = te[k]
        for c in range(Gtr.shape[1]):
            if Gtr[:, c].mean() <= 0.01:
                continue
            ytr, yte = Gtr[:, c] > 0.5, Gte[:, c] > 0.5
            if ytr.all() or (~ytr).all():
                continue                                   # always/never: not counted in (iii)
            crisp_te = (Gte[:, c] > 0.9) | (Gte[:, c] < 0.1); crisp_tr = (Gtr[:, c] > 0.9) | (Gtr[:, c] < 0.1)
            for d in (4, 5, 6):
                t = DecisionTreeClassifier(max_depth=d, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Ftr, ytr)
                p = t.predict(Fte)
                res.setdefault(f"d{d}", []).append(bal(p, yte))
                res.setdefault(f"d{d} crisp-eval", []).append(bal(p[crisp_te], yte[crisp_te]))
                if d == 4:
                    t2 = DecisionTreeClassifier(max_depth=d, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Ftr[crisp_tr], ytr[crisp_tr])
                    p2 = t2.predict(Fte)
                    res.setdefault("d4 crisp-train+eval", []).append(bal(p2[crisp_te], yte[crisp_te]))
    return {k: float(np.mean(np.array(v) >= 0.9)) for k, v in res.items()}, len(res["d4"])

if __name__ == "__main__":
    S = ROOT / "results/glass/atlas/spd"
    nets = {"zoo_argmedian_8_5": S / "spd_zoo_argmedian_8_5_original_30000_512.pt", "zoo_argmedian_8_7": S / "spd_zoo_argmedian_8_7_original_30000_512.pt",
            "zoo_argmedian_16_5": S / "spd_zoo_argmedian_16_5_original_30000_512.pt", "zoo_argmedian_16_7": S / "spd_zoo_argmedian_16_7_original_30000_512.pt",
            "zoo_2nd_argmax_16_10": S / "spd_zoo_2nd_argmax_16_10_original_30000_2048.pt", "zoo_2nd_argmax_8_8": S / "spd_zoo_2nd_argmax_8_8_original_30000_512.pt",
            "zoo_2nd_argmax_8_5": ROOT / "results/glass/units/u19c_model_zoo_2nd_argmax_8_5_original_30000_512.pt"}
    for t in sys.argv[1:] or nets:
        r, n = sweep(t, nets[t])
        print(t, f"({n} non-always mechanisms):", ", ".join(f"{k} {v:.2f}" for k, v in r.items()), flush=True)
