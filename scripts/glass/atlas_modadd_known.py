"""A35 known answer (pre-registered DIARY 11:12): among trusted triggers that use a Fourier feature, >= 80% use one of
the key frequencies named in the model's meta (modadd_p113: 8, 11, 22, 23, 25). Chance: 5/56. Also reports, per
matrix, which frequencies the trusted triggers use, and the certificate criteria."""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
t = sys.argv[1] if len(sys.argv) > 1 else "modadd_p113"
a = json.loads((ROOT / f"results/glass/atlas/{t}_d4.json").read_text())
KEY = {8, 11, 22, 23, 25}
n_f, n_key, per = 0, 0, {}
for k, m in a["matrices"].items():
    c = Counter()
    for x in m["mechanisms"]:
        if x["trigger"]["rule"] in ("always", "never") or atlas._conf(x) < atlas.TRUSTED:
            continue
        fr = {int(f) for u in x["trigger"].get("features_used", []) for f in re.findall(r"(?:cos|sin)\((\d+)·", u)}
        if fr:
            n_f += 1; n_key += bool(fr & KEY); c.update(fr)
    per[k] = c.most_common(6)
share = n_key / n_f if n_f else float("nan")
print(f"Trusted triggers using a Fourier feature: {n_f}; using a key frequency: {n_key} ({100 * share:.0f}%; bar 80%, chance {100 * 5 / 56:.0f}%)")
for k, c in per.items():
    if c:
        print(f"  {k}: {c}")
print("KNOWN ANSWER:", "PASS" if n_f and share >= 0.8 else "FAIL")
