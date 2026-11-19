"""Test for CHANGES A11 (comparing atlases)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import atlas
if not hasattr(atlas, "compare_atlases"):
    check(False, "atlas has compare_atlases"); print(f"{len(fails)} FAILED"); sys.exit(1)
a = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_8_5_d4.json").read_text())
c = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_8_5_shuffled_d4.json").read_text())
self_cmp = atlas.compare_atlases(a, a); print(self_cmp["summary"])
check(self_cmp["share_a"] == 1.0 and all(p["a"] == p["b"] for p in self_cmp["pairs"]), "an atlas matches itself fully, mechanism to itself")
ctrl_cmp = atlas.compare_atlases(a, c); print(ctrl_cmp["summary"])
check(ctrl_cmp["share_a"] < self_cmp["share_a"], f"trained vs shuffled control recurs less than self ({ctrl_cmp['share_a']:.2f})")
# A24: every pair names its basis; ARC's M16,10 vs zoo seed 0
m = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_16_10_d4.json").read_text())
s0 = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_16_10_s0_d4.json").read_text())
r = atlas.compare_atlases(m, s0); print(r["summary"]); print(r["pairs"])
check(all(p.get("basis") in ("role and trigger variables", "role only") for p in r["pairs"]) and "role only" in r["summary"], "A24: each pair names its basis, and the summary counts both kinds")
check(all(p["basis"] == "role only" or p["variables"] for p in r["pairs"]), "A24: a 'role and trigger variables' match shares at least one variable")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
