"""Test for CHANGES A15 (gate v2). Fails before (no v2 variant, no off_recall), passes after.
Known answer: the planted RNN, whose triggers are exact. A conservative gate (switch off only where the trigger's
P(needed) < GATE_TAU) must still switch its mechanisms off where they are not needed (off_recall >= 0.9) and keep the
network's decisions (fidelity >= 0.99). Design case: on M16,10, v2 fidelity must exceed v1."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)

check(getattr(atlas, "GATE_TAU", None) == 0.05, f"GATE_TAU is the pre-registered 0.05 (got {getattr(atlas, 'GATE_TAU', None)})")
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
f, g = a["fidelity"], a.get("gate", {})
check("rule_gated_trusted_v2" in f, "planted: fidelity has rule_gated_trusted_v2")
check(f.get("rule_gated_trusted_v2", 0) >= 0.99, f"planted: v2 fidelity >= 0.99 (got {f.get('rule_gated_trusted_v2')})")
check(g.get("v2", {}).get("off_recall", 0) >= 0.9, f"planted: v2 off_recall >= 0.9 (got {g.get('v2', {}).get('off_recall')})")
check(g.get("v1", {}).get("off_recall", 0) >= g.get("v2", {}).get("off_recall", 1), "planted: v1 switches off at least as often as v2")
for v in ("v1", "v2"):
    x = g.get(v, {})
    check(0 <= x.get("off_share", -1) <= 1 and 0 <= x.get("off_recall", -1) <= 1, f"planted: {v} off_share and off_recall are shares")
check(f.get("rule_gated_trusted", 0) >= 0.99, f"planted: v1 fidelity kept (got {f.get('rule_gated_trusted')})")
m = atlas.build_atlas(atlas.algzoo_rnn("zoo_2nd_argmax_16_10"), spd_model=ROOT / "results/glass/atlas/spd/spd_zoo_2nd_argmax_16_10_original_30000_2048.pt", depth=4)
fm = m["fidelity"]
check(abs(fm["rule_gated_trusted"] - 0.700) < 0.0015, f"M16,10: v1 fidelity unchanged (0.700; got {fm['rule_gated_trusted']:.4f})")
check(fm.get("rule_gated_trusted_v2", 0) > fm["rule_gated_trusted"], f"M16,10: v2 fidelity > v1 (got {fm.get('rule_gated_trusted_v2')})")
print("M16,10 gate:", m.get("gate"), "v2 fidelity", fm.get("rule_gated_trusted_v2"), "task rule", m["task_rule"])
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
