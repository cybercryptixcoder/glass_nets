"""Test for CHANGES A29 (what a mechanism reads and writes). Fails before (no `reads`), passes after."""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = atlas.build_atlas(atlas.planted_rnn(), spd_model=ROOT / "results/glass/units/u19b_model_20_40000_3e-3_1.0_1.0_1e-3_512.pt", depth=3, n=16000)
atlas.gloss(a)
for x in a["matrices"]["W_h"]["mechanisms"]:
    r, w = x.get("reads"), x.get("writes")
    if r is None:
        check(False, f"{x['id']}: has reads/writes"); continue
    j = int(np.argmax(np.abs(x["V"])))
    check(r[0]["unit"] == j and r[0]["share"] >= 0.9 and w[0]["unit"] == int(np.argmax(np.abs(x["U"]))) and w[0]["share"] >= 0.9,
          f"{x['id']}: reads and writes its own neuron ({r[0]}, {w[0]})")
m = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_16_10_d4.json").read_text()); atlas.gloss(m)
x = m["matrices"]["W_h"]["mechanisms"][0]
V = np.abs(np.array(x["V"])); sh = V / V.sum()
check(all(abs(e["share"] - sh[e["unit"]]) < 1e-9 for e in x.get("reads", [])) and x.get("reads"), f"M16,10 {x['id']}: shares match the stored V ({x.get('reads')})")
print("example:", x["id"], x.get("rw_text"))
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
