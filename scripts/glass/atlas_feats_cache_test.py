"""Test for CHANGES A36 (feature cache for transformer targets). Fails before (no cache), passes after."""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
t = atlas.make_target("modadd_p113"); x = t.X[:2000]
check(hasattr(t, "_fcache"), "ModAddTarget has a feature cache")
n1, F1 = t._feats(x, "W_in"); n2, F2 = t._feats(x, "W_in")
check(n1 == n2 and np.array_equal(F1, F2), "cached features equal the first computation")
mods = torch.nn.ModuleDict({k: atlas.spd.Decomposed(v, 4).double() for k, v in t.weights.items()})
g = lambda k, n_, F_, a_: np.ones((len(F_), 4))      # noqa: E731
t0 = time.time(); [t.run_gated(mods, x, g) for _ in range(5)]; dt = time.time() - t0
t1 = atlas.make_target("modadd_p113"); t1._feats = lambda x_, name: atlas.ModAddTarget._feats_uncached(t1, x_, name)
t0 = time.time(); [t1.run_gated(mods, x, g) for _ in range(5)]; du = time.time() - t0
print(f"5 gated runs: cached {dt:.1f}s, uncached {du:.1f}s ({du / dt:.1f}x)")
check(dt < du, f"the cache makes gated runs faster ({du / dt:.1f}x; the entry's guess of 5x was wrong, see CHANGES)")
p = atlas.planted_transformer()
a = atlas.build_atlas(p, spd_model=ROOT / "results/glass/atlas/spd/spd_planted_transformer_original_30000_512.pt", depth=4)
check(a["fidelity"]["rule_gated_trusted_v3"] >= 0.99 and abs(a["fidelity"]["mechanisms_alone"] - 1.0) < 1e-9, f"planted transformer numbers unchanged ({a['fidelity']['rule_gated_trusted_v3']}, {a['fidelity']['mechanisms_alone']})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
