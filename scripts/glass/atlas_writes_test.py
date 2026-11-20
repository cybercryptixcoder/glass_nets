"""A44 test (written before the code): the write frequency of an answer-space pattern, in the right basis."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
ok = True
def check(c, m):
    global ok
    print(("PASS " if c else "FAIL ") + m); ok &= bool(c)
if not hasattr(atlas, "answer_wave"):
    check(False, "atlas has answer_wave"); print("SOME FAILED"); sys.exit(1)
p = 113; c = np.arange(p); L = atlas._dlog_table(p)[:p]
w = np.cos(2 * np.pi * 8 * c / p + 0.4)
f, basis, share = atlas.answer_wave(w, p)
check((f, basis) == (8, "line") and share > 0.9, f"number-line wave at 8 -> {f}, {basis}, share {share:.2f}")
w2 = np.where(c > 0, np.cos(2 * np.pi * 4 * np.maximum(L, 0) / (p - 1) + 1.1), 0.0)
f, basis, share = atlas.answer_wave(w2, p)
check((f, basis) == (4, "log") and share > 0.9, f"discrete-log wave at 4 -> {f}, {basis}, share {share:.2f}")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
