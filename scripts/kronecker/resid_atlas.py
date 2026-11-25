"""Where the zeros left by P4 live, on any n (run on held-out n only after the KR1b (B) rule was frozen,
04:01:50 ET). Families as in dev_resid_p4.py. Usage: resid_atlas.py n...  -> results/kronecker/resid_atlas.json"""
import sys, os, json, time
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_reduce as fr
from gwxz import gwxz_zero
from fathook import fathook_zero
from skew import conj
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t) and not fathook_zero(A, B, t)
fr.p2.cache_clear()
from kron_feats import load, b1_masks
from b2_lgbm_lib import orbit_ids
def is_rect(p): return len(set(p)) == 1
def rowform(p): return p if len(p) <= 2 else conj(p)
def fam(key, n):
    tw = [rowform(x) for x in key if len(x) <= 2 or x[0] <= 2]
    if tw:
        p = max(x[1] if len(x) > 1 else 0 for x in tw)
        return "two-row, Catalan (k,k)" if (n % 2 == 0 and p == n // 2) else ("two-row, (k+1,k)" if (n % 2 == 1 and p == n // 2) else "two-row, other")
    if any(len(x) > 1 and x[1] == 1 for x in key): return "hook"
    x, y, w = key
    for u, v in ((x, y), (x, w), (y, w)):
        if is_rect(u) and (v == u or v == conj(u)): return "rectangle squared"
    return "other"
out = json.load(open("results/kronecker/resid_atlas.json")) if os.path.exists("results/kronecker/resid_atlas.json") else {}
for n in map(int, sys.argv[1:]):
    t0 = time.time()
    parts, g = load(n); P = len(parts); Pp = [tuple(int(x) for x in p if x) for p in parts]
    nz = (g != 0).ravel(); ok = b1_masks(parts).ravel(); orb = orbit_ids(parts).ravel(); del g
    uo, inv = np.unique(orb, return_inverse=True); cnt = np.bincount(inv.ravel()); del orb, inv
    u, rem = np.divmod(uo, P * P); v, w = np.divmod(rem, P)
    c = Counter(); norb = Counter()
    for i in np.flatnonzero(ok[uo] & ~nz[uo]):
        T = (Pp[u[i]], Pp[v[i]], Pp[w[i]])
        if fr.p2(*fr.canon(*T)):
            f = fam(T, n); c[f] += int(cnt[i]); norb[f] += 1
    tot = sum(c.values())
    out[str(n)] = dict(n=n, p4_missed_zeros=tot, triples=int(nz.size), by_family={k: [v, v / tot] for k, v in sorted(c.items())},
                       orbits=dict(norb), seconds=round(time.time() - t0))
    print(out[str(n)], flush=True)
    json.dump(out, open("results/kronecker/resid_atlas.json", "w"), indent=1)
