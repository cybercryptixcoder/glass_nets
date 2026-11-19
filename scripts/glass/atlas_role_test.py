"""Test for CHANGES A7 (task-relative effects)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import atlas
check(hasattr(atlas, "_scalar_class_role"), "atlas has _scalar_class_role")
if not fails:
    x = np.array([[0.1, 0.9, -0.5, 0.4, 0.2]])[:, :, None]
    check(atlas._scalar_class_role(x).tolist() == [[4, 1, 5, 2, 3]], f"ranks of a hand-made input ({atlas._scalar_class_role(x).tolist()})")
    a = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_8_5_d4.json").read_text())
    ok = all("role_changes" in x["effect"] for m in a["matrices"].values() for x in m["mechanisms"])
    check(ok, "every M8,5 mechanism effect has role changes")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
