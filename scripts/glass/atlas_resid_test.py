"""Test for CHANGES A17: the residual judged by what it does (the mechanisms alone). Fails before (no field), passes after."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_transformer(), spd_model=ROOT / "results/glass/atlas/spd/spd_planted_transformer_original_30000_512.pt", depth=4)
ma = a["fidelity"].get("mechanisms_alone")
check(ma is not None and ma >= 0.99, f"planted transformer: mechanisms alone >= 0.99 (got {ma})")
for k in ("L0_W_V", "L0_W_O", "W_U"):
    m = a["matrices"][k]
    check(m["alive_only_rel_error"] > 0.4, f"planted transformer {k}: weight-space error > 0.4 ({m['alive_only_rel_error']:.3f}; on data {m.get('alive_only_rel_error_on_data', float('nan')):.3f}; mechanisms alone in this matrix {m.get('mechanisms_alone_fidelity')})")
r = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
check(r["fidelity"].get("mechanisms_alone", 0) >= 0.99, f"planted RNN: mechanisms alone >= 0.99 (got {r['fidelity'].get('mechanisms_alone')})")
m = atlas.build_atlas(atlas.algzoo_rnn("zoo_2nd_argmax_8_5"), spd_model=ROOT / "results/glass/units/u19c_model_zoo_2nd_argmax_8_5_original_30000_512.pt", depth=4)
check(abs(m["fidelity"].get("mechanisms_alone", 0) - 0.9805) < 0.00051, f"M8,5: mechanisms alone reproduces A12's 'before' all-alive 0.9805 (got {m['fidelity'].get('mechanisms_alone')})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
