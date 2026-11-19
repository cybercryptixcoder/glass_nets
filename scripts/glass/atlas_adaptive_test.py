"""Test for CHANGES A28 part 2 (adaptive decomposition). Fails before (no build_adaptive), passes after.
The choice must be the first rung (sparsest first) whose TRAINING-half numbers clear the bar, and must not depend on
the held-out half: corrupting the held-out half's labels does not change the choice."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
if not hasattr(atlas, "build_adaptive"):
    check(False, "atlas has build_adaptive"); print(f"{len(fails)} FAILED"); sys.exit(1)
S = ROOT / "results/glass/atlas/spd"
ladder = [S / "spd_zoo_argmedian_16_5_original_30000_512.pt", S / "spd_zoo_argmedian_16_5_original_30000_512_mse3e-4.pt", S / "spd_zoo_argmedian_16_5_original_30000_512_mse1e-4.pt"]
a = atlas.build_adaptive(atlas.algzoo_rnn("zoo_argmedian_16_5"), ladder)
log = a["adaptive"]["ladder"]
first_ok = next((i for i, r in enumerate(log) if r["train_alone"] >= r["bar"] and r["train_v3"] >= r["bar"]), len(log) - 1)
check(log[first_ok]["chosen"] and len(log) == first_ok + 1, f"chooses the first rung that clears the training-half bar (rung {first_ok}; {[(round(r['train_alone'], 3), round(r['train_v3'], 3)) for r in log]})")
tgt = atlas.algzoo_rnn("zoo_argmedian_16_5"); X = tgt.X[:16000]; h = len(X) // 2
tgt.X = X.copy(); tgt.X[h:] = tgt.X[h:][::-1].copy() * 0 + tgt.X[h:].mean()     # wreck the held-out half
b = atlas.build_adaptive(tgt, ladder)
check([r["chosen"] for r in b["adaptive"]["ladder"]] == [r["chosen"] for r in log], "the choice does not depend on the held-out half")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
