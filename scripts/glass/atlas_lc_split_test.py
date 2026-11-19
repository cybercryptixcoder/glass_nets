"""Test for CHANGES A32 (transformer domain rows shuffled at load). Fails before, passes after."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
for t in ("lc_3_4", "lc_4_5", "lc_6_5"):
    tok = atlas.make_target(t).X.argmax(-1); h = len(tok) // 2
    d = abs(tok[:h, 0].mean() - tok[h:, 0].mean())
    check(d < 0.2, f"{t}: first-token means of the two halves differ by {d:.2f} (< 0.2)")
p = atlas.planted_transformer().X.argmax(-1)
check(p.shape == (16000, 4), "planted transformer unchanged in shape")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
