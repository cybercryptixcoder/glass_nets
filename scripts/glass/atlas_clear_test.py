"""Test for CHANGES A19 (trigger confidence on clear-cut cases). Fails before (no confidence_clear), passes after."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
cc = [x["trigger"].get("confidence_clear") for m in a["matrices"].values() for x in m["mechanisms"]]
check(all(c is not None and c >= 0.95 for c in cc), f"planted: every trigger confidence_clear >= 0.95 ({cc})")
gs = [x["trigger"].get("graded_share") for m in a["matrices"].values() for x in m["mechanisms"]]
check(all(g is not None and 0 <= g <= 1 for g in gs), "planted: graded_share is a share")
check(a["fidelity"].get("rule_gated_trusted_v3", 0) >= 0.99, f"planted: v3 fidelity >= 0.99 (got {a['fidelity'].get('rule_gated_trusted_v3')})")
check("v3" in a.get("gate", {}), "planted: gate work measured for v3")
m = atlas.build_atlas(atlas.algzoo_rnn("zoo_2nd_argmax_8_5"), spd_model=ROOT / "results/glass/units/u19c_model_zoo_2nd_argmax_8_5_original_30000_512.pt", depth=4)
check(abs(m["fidelity"]["rule_gated_trusted"] - 0.923) < 0.0006, f"M8,5: v1 unchanged (0.923; got {m['fidelity']['rule_gated_trusted']:.4f})")
check(abs(m["fidelity"]["rule_gated_trusted_v2"] - 0.948) < 0.0006, f"M8,5: v2 unchanged (0.948; got {m['fidelity']['rule_gated_trusted_v2']:.4f})")
print("M8,5 v3", m["fidelity"].get("rule_gated_trusted_v3"), m.get("gate", {}).get("v3"))
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
