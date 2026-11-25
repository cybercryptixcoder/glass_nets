"""Dev only (n <= 14): the zeros P4 misses (ordered triples), by family. Two-row families are split by the
two-row shape (n-p, p) (or its conjugate) with the largest p present: 'cat' means p = floor(n/2)."""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, "scripts/kronecker")
import face_reduce as fr
from gwxz import gwxz_zero
from fathook import fathook_zero
from skew import conj
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t) and not fathook_zero(A, B, t)
fr.p2.cache_clear()
from kron_feats import load, b1_masks
def is_rect(p): return len(set(p)) == 1
def rowform(p): return p if len(p) <= 2 else conj(p)
def fam(key, n):
    tw = [rowform(x) for x in key if len(x) <= 2 or x[0] <= 2]
    if tw:
        p = max(x[1] if len(x) > 1 else 0 for x in tw)
        return f"two-row p={p}" + (" (cat)" if p == n // 2 else "")
    if any(len(x) > 1 and x[1] == 1 for x in key): return "hook"
    x, y, w = key
    for u, v in ((x, y), (x, w), (y, w)):
        if is_rect(u) and (v == u or v == conj(u)): return "rect^2"
    return "other"
show = sys.argv[1] == "1"
for n in map(int, sys.argv[2:]):
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]
    nz = g != 0; ok = b1_masks(parts); seen = {}; cnt = Counter(); orbs = {}
    for a, b, c in np.argwhere(ok & ~nz):
        key = fr.canon(P[a], P[b], P[c])
        if key not in seen:
            seen[key] = fam(key, n) if fr.p2(*key) else None
            if seen[key]: orbs[key] = seen[key]
        if seen[key]: cnt[seen[key]] += 1
    print(f"n={n}: P4 missed {sum(cnt.values())}: {dict(sorted(cnt.items()))}; orbits {dict(Counter(orbs.values()))}", flush=True)
    if show:
        for k in sorted(orbs, key=lambda k: (orbs[k], k)): print("   ", orbs[k], " | ".join("".join(map(str, x)) for x in k))
