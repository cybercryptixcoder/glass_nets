"""Test for CHANGES A28 part 1 (training-half fidelities, for choosing a decomposition). Fails before, passes after."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
ft = a.get("fidelity_train", {})
check(ft.get("rule_gated_trusted_v3", 0) >= 0.99 and ft.get("mechanisms_alone", 0) >= 0.99, f"planted: training-half fidelities >= 0.99 ({ft})")
m = atlas.build_atlas(atlas.algzoo_rnn("zoo_2nd_argmax_8_5"), spd_model=ROOT / "results/glass/units/u19c_model_zoo_2nd_argmax_8_5_original_30000_512.pt", depth=4)
for key, want in (("rule_gated_trusted_v3", 0.9505), ("rule_gated_trusted_v4", None), ("mechanisms_alone", 0.9805)):
    if want is not None:
        check(abs(m["fidelity"][key] - want) < 0.0006, f"M8,5: held-out {key} unchanged ({want}; got {m['fidelity'][key]:.4f})")
check("fidelity_train" in m and abs(m["fidelity_train"]["mechanisms_alone"] - m["fidelity"]["mechanisms_alone"]) < 0.02, f"M8,5: training and held-out 'alone' agree within 0.02 ({m.get('fidelity_train')})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
