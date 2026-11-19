"""A41 known answers (pre-registered DIARY 17:10). Among trusted mechanisms (not always/never) in W_in, W_out and W_U:
K1 (memorizing network, modrand_p113): >= 80% labelled lookup. K2 (structured adders, modadd_p113_s1/_s2): >= 80% labelled
rule. Other matrices are reported, not judged.
Usage: python3 scripts/glass/atlas_lookup_known.py TARGET [ATLAS_JSON] --expect=lookup|rule"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
pos = [x for x in sys.argv[1:] if not x.startswith("--")]
opt = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--"))
t = pos[0]; path = Path(pos[1]) if len(pos) > 1 else ROOT / f"results/glass/atlas/{t}_d4.json"
expect = opt["expect"]
a = json.loads(path.read_text())
JUDGED = ("W_in", "W_out", "W_U")
tot = Counter(); per = {}
for k, m in a["matrices"].items():
    c = Counter(x["trigger"].get("label", "missing") for x in m["mechanisms"]
                if x["trigger"]["rule"] not in ("always", "never") and atlas._conf(x) >= atlas.TRUSTED)
    per[k] = c
    if k in JUDGED:
        tot.update(c)
n = sum(tot.values()); share = tot[expect] / n if n else float("nan")
print(f"{t} ({path.name}): trusted mechanisms in {', '.join(JUDGED)}: {n}; labelled {expect}: {tot[expect]} ({100 * share:.0f}%; bar 80%)  {dict(tot)}")
for k, c in per.items():
    if c:
        print(f"  {k}: {dict(c)}")
print("KNOWN ANSWER:", "PASS" if n and share >= 0.8 else "FAIL")
