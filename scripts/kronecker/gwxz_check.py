import sys, numpy as np
sys.path.insert(0, "scripts/kronecker")
from kron_feats import load
for k in range(2, 8):
    n = 2 * k; parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]; idx = {p: i for i, p in enumerate(P)}
    r = idx[(k, k)]
    sup = [P[c] for c in range(len(P)) if g[r, r, c] != 0]
    four = [p for p in P if len(p) <= 4]
    def par(p):
        pp = list(p) + [0] * (4 - len(p))          # pad to 4 parts with zeros
        return "even" if all(x % 2 == 0 for x in pp) else ("odd" if all(x % 2 == 1 for x in pp) else "mixed")
    from collections import Counter
    cs = Counter(par(p) for p in sup); cf = Counter(par(p) for p in four)
    print(f"k={k}: support {len(sup)}, all have <=4 parts: {all(len(p) <= 4 for p in sup)}; support parity (pad to 4) {dict(cs)}; all <=4-part shapes {dict(cf)}; max g {int(max(g[r, r, idx[p]] for p in sup))}")
