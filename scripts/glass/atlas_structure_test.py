"""A43 test (written before the code): the structure summary from A41 labels."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
ok = True
def check(c, m):
    global ok
    print(("PASS " if c else "FAIL ") + m); ok &= bool(c)
a = json.loads((ROOT / "results/glass/atlas/modadd_p113_s1_d4_a41.json").read_text())
st = atlas.findings(a)["statements"]
s6 = [s for s in st if s.startswith("6.")]
check(len(s6) == 1, "a '6. Structure' statement exists")
txt = s6[0] if s6 else ""
print(txt)
check("W_U: 13 rule" in txt, "W_U reported as rules")
check("W_E: 1 rule, 34 lookup" in txt, "the embedding reported as mostly lookup")
check(all(f"{k} (line" in txt for k in (30, 19, 38)), "s1's key frequencies 30, 19, 38 are among the top rule frequencies")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
