"""Test for CHANGES A8 (atlas findings)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import atlas
if not hasattr(atlas, "findings"):
    check(False, "atlas has findings"); print(f"{len(fails)} FAILED"); sys.exit(1)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3)
used = [len(x["trigger"].get("features_used", [])) for m in a["matrices"].values() for x in m["mechanisms"]]
check(all(u == 1 for u in used), f"planted: every trigger uses exactly one feature ({used})")
F = atlas.findings(a); print("\n".join(F["statements"]))
check(len(F["statements"]) >= 2, "planted: findings stated")
m = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_8_5_d4.json").read_text())
F2 = atlas.findings(m); print("\n".join(F2["statements"]))
check(any("trusted trigger" in s for s in F2["statements"]), "M8,5: trusted-trigger share stated")
check(len(F2["groups"]) >= 1, "M8,5: at least one functional group")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
