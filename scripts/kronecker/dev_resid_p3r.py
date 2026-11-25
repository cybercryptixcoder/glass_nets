"""Dev only (n <= 14): list the symmetry orbits of zeros that P3r misses, with simple structural tags."""
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
def s(p): return "".join(str(x) if x < 10 else f"({x})" for x in p)
def is_rect(p): return len(set(p)) == 1
def tag(key):
    x, y, w = key
    for u, v, o in ((x, y, w), (x, w, y), (y, w, x)):
        if is_rect(u) and v == u: return "RR"
        if is_rect(u) and v == conj(u): return "RR'"
    for u, v, o in ((x, y, w), (x, w, y), (y, w, x)):
        if v == u: return "xx"
        if v == conj(u): return "xx'"
    if any(is_rect(u) for u in key): return "R"
    return "-"
show = int(sys.argv[1]); ns = list(map(int, sys.argv[2:]))
for n in ns:
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]
    nz = g != 0; ok = b1_masks(parts); res = {}
    for a, b, c in np.argwhere(ok & ~nz):
        key = fr.canon(P[a], P[b], P[c])
        if key in res: continue
        if fr.p2(*key): res[key] = tag(key)
    print(f"n={n}: P3r residual orbits {len(res)}; tags {dict(Counter(res.values()))}", flush=True)
    if show:
        for k in sorted(res, key=lambda k: (res[k], k)):
            print(f"   [{res[k]:3s}]", " | ".join(s(x) for x in k))
