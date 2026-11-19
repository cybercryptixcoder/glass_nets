"""Test for CHANGES A31 (read and write profiles by unit label). Fails before, passes after."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
if not hasattr(atlas, "profile"):
    check(False, "atlas.profile exists"); print(f"{len(fails)} FAILED"); sys.exit(1)
labels = {int(k): v.split(": ", 1)[-1] for k, v in json.loads((ROOT / "results/glass/units/maps/zoo_2nd_argmax_16_10.json").read_text())["map"]["per_unit"].items()}
v = np.zeros(16); v[[1, 6, 7]] = [0.5, -0.3, 0.2]
pr = atlas.profile(v, labels)
check(abs(sum(s for _, s in pr["all"]) - 1) < 1e-9 and all(s >= 0 for _, s in pr["all"]), "shares are non-negative and sum to 1")
check(pr["all"][0][0] == labels[1] == labels[6] == labels[7] and abs(pr["all"][0][1] - 1) < 1e-9, f"neurons 1, 6, 7 pool into one label at 100% ({pr['all'][:2]})")
a = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_16_10_d4.json").read_text()); atlas.gloss(a)
x = a["matrices"]["W_h"]["mechanisms"][0]
V = np.abs(np.array(x["V"])); want = {}
for j, w in enumerate(V / V.sum()):
    want[labels.get(j, "unexplained")] = want.get(labels.get(j, "unexplained"), 0) + w
got = dict(x.get("read_profile", {}).get("all", []))
check(got and all(abs(got[k] - want[k]) < 1e-9 for k in want), f"{x['id']}: read profile equals the pooled stored V ({x.get('profile_text')})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
