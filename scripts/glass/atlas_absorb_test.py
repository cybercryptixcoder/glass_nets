"""Test for CHANGES A26 (residual absorption). Fails before (no absorb option), passes after."""
import inspect, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
if "absorb" not in inspect.signature(atlas.build_atlas).parameters:
    check(False, "build_atlas takes absorb"); print(f"{len(fails)} FAILED"); sys.exit(1)
p = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000, absorb=True)
for m in ("W_x", "W_h", "W_o"):
    mech = p["matrices"][m]["mechanisms"]
    check(len(mech) == 5 and min(x["trigger"]["confidence"] for x in mech) >= 0.95, f"planted, absorbed: {m} keeps 5 mechanisms with triggers >= 0.95")
    check(all(x["effect"]["specificity"] >= 10 for x in mech), f"planted, absorbed: {m} specificity >= 10")
check(p["fidelity"]["rule_gated_trusted_v3"] >= 0.99 and p.get("absorb") is True, f"planted, absorbed: v3 >= 0.99 ({p['fidelity']['rule_gated_trusted_v3']:.4f}), flagged absorb")
S = ROOT / "results/glass/atlas/spd/spd_zoo_argmedian_16_5_original_30000_512_mse1e-4.pt"
a = atlas.build_atlas(atlas.algzoo_rnn("zoo_argmedian_16_5"), spd_model=S, depth=4, absorb=True)
check(a["fidelity"]["mechanisms_alone"] >= 0.999, f"argmedian 16_5 at 1e-4, absorbed: mechanisms alone {a['fidelity']['mechanisms_alone']:.4f} (design value 1.000)")
check(a["fidelity"]["rule_gated_trusted_v3"] >= a["task_rule"], f"argmedian 16_5 at 1e-4, absorbed: v3 {a['fidelity']['rule_gated_trusted_v3']:.4f} >= task rule {a['task_rule']:.4f}")
b = atlas.build_atlas(atlas.algzoo_rnn("zoo_argmedian_16_5"), spd_model=ROOT / "results/glass/atlas/spd/spd_zoo_argmedian_16_5_original_30000_512.pt", depth=4)
for key, want in (("rule_gated_trusted_v3", 0.818), ("mechanisms_alone", 0.900)):
    check(abs(b["fidelity"][key] - want) < 0.0006, f"without absorb unchanged: {key} {b['fidelity'][key]:.4f} ({want})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
