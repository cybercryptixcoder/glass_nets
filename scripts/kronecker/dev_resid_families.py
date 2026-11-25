"""Dev only (n <= 14): split the zeros P3r misses (ordered triples) into families.
F1: some argument has at most 2 rows or at most 2 columns (two-row shapes: Rosas, Ballantine-Orellana);
F2: else some argument is a hook (Blasiak, Remmel); F3: else two arguments are the same rectangle up to
conjugation (rectangular Kronecker coefficients); F4: the rest. Also: is a Dvir bound tight (face) or not."""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, "scripts/kronecker")
import face_reduce as fr
from gwxz import gwxz_zero
from skew import conj
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t)
fr.p2.cache_clear()
from kron_feats import load, b1_masks, inter
def is_rect(p): return len(set(p)) == 1
def two_row(p): return len(p) <= 2 or p[0] <= 2
def hook(p): return len(p) > 1 and p[1] == 1
def family(key):
    if any(two_row(x) for x in key): return "F1"
    if any(hook(x) for x in key): return "F2"
    x, y, w = key
    for u, v in ((x, y), (x, w), (y, w)):
        if is_rect(u) and (v == u or v == conj(u)): return "F3"
    return "F4"
def on_face(A, B, t):
    for (x, y, w) in ((A, B, t), (B, t, A), (t, A, B)):
        for (u, v, s) in ((x, y, w), (conj(x), conj(y), w), (conj(x), y, conj(w)), (x, conj(y), conj(w))):
            if s[0] == sum(min(a, b) for a, b in zip(u, v)): return True
    return False
def main():
  for n in map(int, sys.argv[1:]):
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]
    nz = g != 0; ok = b1_masks(parts); seen = {}; cnt = Counter(); face = Counter(); orbits = Counter()
    for a, b, c in np.argwhere(ok & ~nz):
        key = fr.canon(P[a], P[b], P[c])
        if key not in seen:
            seen[key] = (fr.p2(*key), family(key), on_face(*key))
            if seen[key][0]: orbits[seen[key][1]] += 1
        miss, fam, f = seen[key]
        if miss: cnt[fam] += 1; face[(fam, f)] += 1
    tot = sum(cnt.values())
    print(f"n={n}: P3r missed zeros {tot}; by family (ordered) {dict(sorted(cnt.items()))}; orbits {dict(sorted(orbits.items()))}; on a tight face {dict(sorted(face.items()))}", flush=True)

if __name__ == "__main__":
    main()
