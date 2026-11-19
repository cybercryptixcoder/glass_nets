"""Test for CHANGES A20 (trigger glosses from unit maps). Fails before (no gloss), passes after."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
if not hasattr(atlas, "gloss"):
    check(False, "atlas.gloss exists"); print(f"{len(fails)} FAILED"); sys.exit(1)
a = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_16_10_d4.json").read_text())
labels = json.loads((ROOT / "results/glass/units/maps/zoo_2nd_argmax_16_10.json").read_text())["map"]["per_unit"]
g = atlas.gloss(a)
n_gl, n_un = 0, 0
for m in g["matrices"].values():
    for x in m["mechanisms"]:
        r, rg = x["trigger"]["rule"], x["trigger"].get("rule_glossed")
        check(rg is not None, f"{x['id']}: has rule_glossed") if rg is None else None
        if rg is None:
            continue
        check(re.sub(r" \[≈ [^\]]*\]", "", rg) == r, f"{x['id']}: removing the glosses gives back the rule") if re.sub(r" \[≈ [^\]]*\]", "", rg) != r else None
        for j in set(re.findall(r"neuron (\d+)", r)):
            lab = labels.get(j, "unexplained")
            has = f"neuron {j} [≈" in rg
            if lab == "unexplained":
                n_un += 1; check(not has, f"{x['id']}: unexplained neuron {j} has no gloss") if has else None
            else:
                n_gl += 1; check(has, f"{x['id']}: neuron {j} carries its label") if not has else None
check(n_gl > 0, f"glossed neuron mentions: {n_gl}; unexplained left bare: {n_un}")
ex = next(x for m in g["matrices"].values() for x in m["mechanisms"] if "[≈" in x["trigger"].get("rule_glossed", ""))
print("example:", ex["id"], ex["trigger"]["rule_glossed"][:300])
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
