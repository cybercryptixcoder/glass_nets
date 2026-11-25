"""Dev only (n <= 14): print P3r residue orbits of one family, normalised so the two-row (or hook) shape is
written in row form first. Usage: dev_resid_view.py FAMILY n..."""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, "scripts/kronecker")
import face_reduce as fr
from gwxz import gwxz_zero
from skew import conj
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t)
fr.p2.cache_clear()
from kron_feats import load, b1_masks
from dev_resid_families import family, two_row, hook
def s(p): return "".join(str(x) if x < 10 else f"({x})" for x in p)
def normal(key):
    """Rotate/conjugate (even number of conjugations) so the first entry is the special shape in row form."""
    x, y, w = key
    best = None
    for a, b, c in ((x, y, w), (y, w, x), (w, x, y), (x, w, y), (y, x, w), (w, y, x)):
        for (u, v, t) in ((a, b, c), (conj(a), conj(b), c), (conj(a), b, conj(c)), (a, conj(b), conj(c))):
            sc = (0 if (len(u) <= 2) else 1 if hook(u) else 2, len(u), u, v, t)
            if best is None or sc < best: best = sc
    return best[2:]
fam = sys.argv[1]
for n in map(int, sys.argv[2:]):
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]
    nz = g != 0; ok = b1_masks(parts); res = set(); seen = set()
    for a, b, c in np.argwhere(ok & ~nz):
        key = fr.canon(P[a], P[b], P[c])
        if key in seen: continue
        seen.add(key)
        if fr.p2(*key) and family(key) == fam: res.add(normal(key))
    print(f"n={n} family {fam}: {len(res)} orbits")
    for k in sorted(res): print("   ", " | ".join(s(x) for x in k))
