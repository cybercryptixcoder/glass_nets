import sys, numpy as np
sys.path.insert(0, "scripts/kronecker")
import face_reduce as fr
from gwxz import gwxz_zero
from functools import lru_cache
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t)     # GWXZ inside the recursion too
fr.p2.cache_clear()
from kron_feats import load, b1_masks
for n in map(int, sys.argv[1:]):
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]
    nz = g != 0; ok = b1_masks(parts); seen = {}; miss = fz = 0
    for a, b, c in np.argwhere(ok):
        key = fr.canon(P[a], P[b], P[c])
        if key not in seen: seen[key] = fr.p2(*key)
        v = seen[key]
        if nz[a, b, c]: fz += (not v)
        else: miss += v
    print(f"n={n}: P3r missed zeros {miss}, false zeros {fz}, accuracy {1 - (miss + fz) / nz.size:.5f}", flush=True)
