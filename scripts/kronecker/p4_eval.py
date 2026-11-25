"""P4 = P3r (P2 + GWXZ) with the Berele-Regev fat-hook bound in the zero test, inside the recursion too.
Usage: p4_eval.py n...   (dev n only until the KR1 held-out scoring)"""
import sys, numpy as np
sys.path.insert(0, "scripts/kronecker")
import face_reduce as fr
from gwxz import gwxz_zero
from fathook import fathook_zero
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t) and not fathook_zero(A, B, t)
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
    print(f"n={n}: P4 missed zeros {miss}, false zeros {fz}, accuracy {1 - (miss + fz) / nz.size:.6f}", flush=True)
