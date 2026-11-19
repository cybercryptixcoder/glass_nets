"""Test for CHANGES A1 (mechanism atlas v1). Fails before (no module), passes after.
Known answer: the planted RNN (sum of 5 sparse features over 6 steps; W_x = W_h = W_o = I), using the saved U19(b')
decomposition. The atlas must report 5 mechanisms per matrix, every trigger at held-out balanced accuracy >= 0.95, and
rule-gated fidelity >= 0.99."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
try:
    import atlas
except Exception as e:  # noqa: BLE001
    check(False, f"lab/atlas.py imports ({type(e).__name__}: {e})")
    print(f"{len(fails)} FAILED"); sys.exit(1)
tgt = atlas.planted_rnn()
a = atlas.build_atlas(tgt, spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
for m in ("W_x", "W_h", "W_o"):
    mech = a["matrices"][m]["mechanisms"]
    check(len(mech) == 5, f"{m}: 5 mechanisms (got {len(mech)})")
    worst = min((x["trigger"]["confidence"] for x in mech), default=0)
    check(worst >= 0.95, f"{m}: every trigger at held-out balanced accuracy >= 0.95 (worst {worst:.3f})")
for m in ("W_x", "W_h", "W_o"):          # A2: effects (planted mechanism j acts on output j only)
    for x in a["matrices"][m]["mechanisms"]:
        e = x.get("effect")
        if e is None:
            check(False, f"{x['id']}: has an effect"); break
        planted = int(np.argmax(np.abs(np.array(x["U"] if m != "W_o" else x["U"]))))   # the planted feature index
        check(e["most_lowered_class"] == (planted if m == "W_o" else int(np.argmax(np.abs(np.array(x["U"]))))), f"{x['id']}: removing it lowers its own feature's output (class {e['most_lowered_class']})")
        check(e["specificity"] >= 10, f"{x['id']}: specificity >= 10 (got {e['specificity']:.1f})")
check(a["fidelity"]["rule_gated"] >= 0.99, f"rule-gated fidelity >= 0.99 (got {a['fidelity']['rule_gated']:.4f})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
