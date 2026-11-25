"""Independent checks of the exact tables (used as ground truth by every KR1 score).
For each n: (a) g is symmetric in its three arguments; (b) g(l,m,v) = g(l',m',v) (conjugating two arguments);
(c) g((n),m,v) = [m = v]; (d) g((n-1,1),m,m) = (number of distinct parts of m) - 1;
(e) sum_v g(l,m,v) f^v = f^l f^m for every pair (l, m). Usage: table_checks.py n..."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kron_feats import load, conj
res = {}
for n in map(int, sys.argv[1:]):
    z = np.load(os.path.join(os.environ.get("KDATA", "/home/claude/kdata"), f"kron_n{n}.npz"), allow_pickle=False)
    parts, g, dims = z["parts"], z["g"], z["dims"].astype(np.int64)
    P = [tuple(int(x) for x in p if x) for p in parts]; idx = {p: i for i, p in enumerate(P)}
    cj = np.array([idx[tuple(int(x) for x in q if x)] for q in conj(parts)])
    r = {}
    r["sym_01"] = bool((g == g.transpose(1, 0, 2)).all()); r["sym_02"] = bool((g == g.transpose(2, 1, 0)).all())
    r["conj_two"] = bool((g == g[cj][:, cj]).all())
    t = idx[(n,)]; r["trivial"] = bool((g[t] == np.eye(len(P), dtype=g.dtype)).all())
    if n >= 2:
        s = idx[(n - 1, 1)]
        r["standard_diag"] = bool(all(int(g[s, i, i]) == len(set(p)) - 1 for i, p in enumerate(P)))
    lhs = np.einsum("lmv,v->lm", g.astype(np.int64), dims); rhs = np.outer(dims, dims)
    r["dimension_identity_all_pairs"] = bool((lhs == rhs).all())
    r["all_pass"] = all(r.values())
    res[n] = r; print(n, r, flush=True)
    del g, lhs
os.makedirs("results/kronecker", exist_ok=True)
old = {}
p = "results/kronecker/table_checks.json"
if os.path.exists(p): old = json.load(open(p))
old.update({str(k): v for k, v in res.items()})
json.dump(old, open(p, "w"), indent=1)
