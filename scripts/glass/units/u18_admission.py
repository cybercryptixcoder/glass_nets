"""U18: replace ONLY substitution-grade units (relative error <= eps95), all at once; is fidelity >= the task rule?
(pre-registered in DIARY)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab")); sys.path.insert(0, str(Path(__file__).parent))
import judge, unitmap
from u17_replace_curve import unit_fn

out = {}
for tid in sys.argv[1:]:
    f = ROOT / f"results/glass/units/maps/{tid}.json"
    m = json.loads(f.read_text())["map"] if f.exists() else unitmap.unit_map(tid)
    meta, W, X = judge.load_target(tid); X = np.asarray(X[:8000], dtype=np.float64); Xt = X[4000:]
    acc = float(meta["test_acc"]); later_tot = sum(c["var"] for c in m["cells"] if c["t"] > 1) or 1.0
    sg = [c for c in m["cells"] if c.get("substitution_grade") and c["prims"]]
    rep = {(c["t"], c["unit"]): unit_fn(c) for c in sg}
    fid = float(judge.same(unitmap.run(W, Xt), unitmap.run(W, Xt, rep)).mean())
    share = sum(c["var"] for c in sg if c["t"] > 1) / later_tot
    out[tid] = {"acc": acc, "eps95": m["eps95"], "n_grade": len(sg), "n_cells": len(m["cells"]), "share_after_step1": share, "fidelity": fid, "pass": fid >= acc}
    print(f"{tid}: acc {acc:.3f} eps95 {m['eps95']:.4f} | {len(sg)}/{len(m['cells'])} substitution-grade ({100*share:.0f}% of step>=2 variance) | joint fidelity {fid:.3f} -> {'PASS' if fid >= acc else 'fail'}", flush=True)
(ROOT / f"results/glass/units/u18_{len(out)}.json").write_text(json.dumps(out, indent=1))
