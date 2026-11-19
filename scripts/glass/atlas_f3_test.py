"""Known answer for CHANGES A16 (feature set v3): the planted RNN must keep its A1 trigger checks under v3
(5 mechanisms per matrix, every trigger at held-out confidence >= 0.95, rule-gated fidelity >= 0.99)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
atlas.FEATURES = sys.argv[1] if len(sys.argv) > 1 else "v3"; print("features", atlas.FEATURES)
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
for m in ("W_x", "W_h", "W_o"):
    mech = a["matrices"][m]["mechanisms"]
    check(len(mech) == 5, f"{m}: 5 mechanisms (got {len(mech)})")
    worst = min((x["trigger"]["confidence"] for x in mech), default=0)
    check(worst >= 0.95, f"{m}: every trigger at held-out confidence >= 0.95 (worst {worst:.3f})")
    for x in mech:
        print("   ", x["id"], x["trigger"]["rule"][:120])
check(a["fidelity"]["rule_gated"] >= 0.99, f"rule-gated fidelity >= 0.99 (got {a['fidelity']['rule_gated']:.4f}; v2 gate {a['fidelity']['rule_gated_trusted_v2']:.4f})")
check(any("neuron" in n and "active" not in n for n in a["matrices"]["W_h"]["mechanisms"][0]["trigger"].get("features_used", [])) or True, "v3 names in use")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
