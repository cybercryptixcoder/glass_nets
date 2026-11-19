"""Test for CHANGES A23 (decision-checked gate, v4). Fails before (no v4), passes after."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
f, g = a["fidelity"], a.get("gate", {})
check(f.get("rule_gated_trusted_v4", 0) >= 0.99, f"planted: v4 fidelity >= 0.99 (got {f.get('rule_gated_trusted_v4')})")
check(g.get("v4", {}).get("off_recall", 0) >= 0.9, f"planted: v4 off_recall >= 0.9 (got {g.get('v4')})")
mech = [x for m in a["matrices"].values() for x in m["mechanisms"]]
check(all(x["trigger"].get("switched_in_certificate") for x in mech if x["trigger"].get("confidence_clear", 0) >= 0.9 and x["trigger"]["rule"] not in ("always", "never")),
      "planted: every trusted mechanism is switched in the certificate")
S = ROOT / "results/glass/atlas/spd"
b = atlas.build_atlas(atlas.algzoo_rnn("zoo_argmedian_16_5"), spd_model=S / "spd_zoo_argmedian_16_5_original_30000_512.pt", depth=4)
w8 = [x for x in b["matrices"]["W_x"]["mechanisms"] if x["id"] == "W_x.8"][0]
check(w8["trigger"].get("switched_in_certificate") is False, f"argmedian 16_5: W_x.8 not switched (single cost {w8['trigger'].get('gating_cost_train')})")
for key, want in (("rule_gated_trusted", 0.732), ("rule_gated_trusted_v2", 0.839), ("rule_gated_trusted_v3", 0.818)):
    check(abs(b["fidelity"][key] - want) < 0.0006, f"argmedian 16_5: {key} unchanged ({want}; got {b['fidelity'][key]:.4f})")
print("argmedian 16_5 v4", b["fidelity"].get("rule_gated_trusted_v4"), b["gate"].get("v4"), "task", b["task_rule"])
m = atlas.build_atlas(atlas.algzoo_rnn("zoo_2nd_argmax_16_10"), spd_model=S / "spd_zoo_2nd_argmax_16_10_original_30000_2048.pt", depth=4)
check(m["fidelity"].get("rule_gated_trusted_v4", 0) >= m["fidelity"]["rule_gated_trusted_v3"], f"M16,10: v4 >= v3 (v4 {m['fidelity'].get('rule_gated_trusted_v4')}, v3 {m['fidelity']['rule_gated_trusted_v3']:.4f})")
print("M16,10 v4", m["fidelity"].get("rule_gated_trusted_v4"), m["gate"].get("v4"), "task", m["task_rule"])
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
