"""New claim of the linear-positions construction, tested: slide month1's position continuously at depth 8.

The construction says the late layers read month1 only through one scalar (its position on a January-December line),
stored at the year and month2 tokens. So writing a different value t of that scalar into the network (along the
direction the judge's map gives it, everything else untouched) must change the decision exactly as model() says
for pos1 = t, including values half-way between two months, which no prompt produces.
Run: python3 lab/constructions/_oracle/steer_month_position.py   (prints agreement; writes steer_month_position.json)
"""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import lm, lmjudge, judge as J  # noqa: E402

T = lm.get_task("gpt2_month_rollover")
src = (HERE / "gpt2_month_rollover_linear_positions.py").read_text()
model, align = J.load_construction(src)
ns = {"np": np}; exec(compile(src, "c", "exec"), ns); U, V, C = ns["U"], ns["V"], ns["C"]
X, N = T.X, T.N
out = T.run(np.arange(N))
_, vals = J.run_model(model, X)
sites = ["resid8@year", "resid8@m2"]
A = {s: T.resid(8, np.arange(N), s.split("@")[1]) for s in sites}
maps = {}
for s in sites:                        # the judge's map at each site, from the variables placed there
    vs = [v for v in vals if s in align.get(v, [])]
    Vm = np.concatenate([vals[v] for v in vs], 1); vm = Vm.mean(0); c = A[s].mean(0)
    M, *_ = np.linalg.lstsq(Vm - vm, A[s] - c, rcond=1e-8)
    k = sum(vals[v].shape[1] for v in vs[:vs.index("pos1")])          # pos1's row in the stacked map
    maps[s] = M[k]
rng = np.random.default_rng(0)
B = rng.choice(N, 300, replace=False)
grid = np.linspace(U.min(), U.max(), 23)          # 12 month positions and the points between them
rows, agree, tot = [], 0, 0
for t in grid:
    hook = {s: A[s][B] + (t - vals["pos1"][B, 0])[:, None] * maps[s][None] for s in sites}
    o = T.run(B, hook=hook)
    pred, _ = J.run_model(model, X[B], {"pos1": np.full(len(B), t)})
    clear = lmjudge.confident(o)
    ok = J.same(pred, o)[clear]
    agree += ok.sum(); tot += clear.sum()
    rows.append({"pos1": round(float(t), 3), "network_next_share": round(float((o.argmax(1) == 1).mean()), 3),
                 "construction_next_share": round(float((pred.argmax(1) == 1).mean()), 3), "agreement_clear": round(float(ok.mean()), 3)})
between = [r for i, r in enumerate(rows) if i % 2 == 1]
res = {"prompts": len(B), "grid": rows, "agreement_all": float(agree / tot),
       "agreement_between_months": float(np.mean([r["agreement_clear"] for r in between]))}
(HERE / "steer_month_position.json").write_text(json.dumps(res, indent=1))
print(f"steering month1's position at depth 8 over {len(grid)} values on {len(B)} prompts: construction predicts the network's clear "
      f"decisions in {res['agreement_all']:.1%} of cases ({res['agreement_between_months']:.1%} at positions between two months)")
for r in rows[::2]:
    print(f"  pos1 {r['pos1']:+.2f}: network says next year {r['network_next_share']:.0%}, construction {r['construction_next_share']:.0%}")
