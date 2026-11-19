"""Test for CHANGES A21 (task-level triggers as explanations). Fails before (no task_level), passes after."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
for k in ("W_x", "W_h", "W_o"):
    for x in a["matrices"][k]["mechanisms"]:
        tl = x["trigger"].get("task_level")
        if tl is None:
            check(False, f"{x['id']}: has task_level"); continue
        if k == "W_x":
            check(tl["confidence_clear"] >= 0.9 and tl["rule"] == x["trigger"]["rule"], f"{x['id']}: trusted task-level trigger equal to its rule ({tl['rule']}, {tl['confidence_clear']:.3f})")
        else:
            check(tl["confidence_clear"] < 0.9, f"{x['id']}: task-level trigger NOT trusted (truth not in the task vocabulary; got {tl['confidence_clear']:.3f})")
m = atlas.build_atlas(atlas.algzoo_rnn("zoo_2nd_argmax_8_5"), spd_model=ROOT / "results/glass/units/u19c_model_zoo_2nd_argmax_8_5_original_30000_512.pt", depth=4)
for key, want in (("rule_gated_trusted", 0.9233), ("rule_gated_trusted_v2", 0.9477), ("rule_gated_trusted_v3", 0.9505)):
    check(abs(m["fidelity"][key] - want) < 0.0006, f"M8,5: {key} unchanged ({want}; got {m['fidelity'][key]:.4f})")
mech = [x for mm in m["matrices"].values() for x in mm["mechanisms"] if x["trigger"]["rule"] != "always"]
share = sum(x["trigger"].get("task_level", {}).get("confidence_clear", 0) >= 0.9 for x in mech) / len(mech)
print(f"M8,5: task-level explanations trusted for {share:.2f} of non-always mechanisms")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
