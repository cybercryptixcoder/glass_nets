"""Steer month1 along the UNFITTED calendar-index direction (January = 0 ... December = 11) at its sites and watch the
decision: if the decision runs on the calendar line, pushing month1 "later" (+k months) must raise "next year"
monotonically, and pushing it earlier must lower it.   python3 lab/steer_calendar.py <target> <depth>"""
import json, sys
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import lm  # noqa: E402

tid, L = sys.argv[1], int(sys.argv[2])
T = lm.get_task(tid)
X, N = T.X, T.N
rng = np.random.default_rng(0)
m = X[:, 0].astype(float)
B = rng.choice(N, 400, replace=False)
hook_dirs = {}
for p in ("year", "m2"):
    A = T.resid(L, np.arange(N), p)
    M, *_ = np.linalg.lstsq((m - m.mean())[:, None], A - A.mean(0), rcond=None)
    hook_dirs[p] = (A[B], M[0])                      # activation change per month later
res = {}
for k in (-6, -3, -1, 0, 1, 3, 6):
    o = T.run(B, hook={f"resid{L}@{p}": a + k * d[None] for p, (a, d) in hook_dirs.items()})
    res[k] = {"next_share": round(float((o.argmax(1) == 1).mean()), 3), "mean_margin": round(float((o[:, 1] - o[:, 0]).mean()), 3)}
mono = all(res[a]["mean_margin"] < res[b]["mean_margin"] for a, b in zip(sorted(res), sorted(res)[1:]))
print(tid, f"depth {L}: month1 shifted by k months along the calendar index:",
      " ".join(f"k={k:+d}: next {v['next_share']:.0%} (margin {v['mean_margin']:+.2f})" for k, v in res.items()), "| monotone:", mono)
(T.dir / f"steer_calendar_depth{L}.json").write_text(json.dumps({"depth": L, "shifts": res, "monotone": mono}, indent=1))
