import sys, time, numpy as np
sys.path.insert(0, "scripts/kronecker")
from kron_feats import load, b1_masks
from face_reduce import p2, canon
for n in map(int, sys.argv[1:]):
    t0 = time.time()
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]
    nz = g != 0; ok = b1_masks(parts); hard = ok & ~nz
    caught = fz = 0; seen = {}
    idx = np.argwhere(ok)
    for a, b, c in idx:
        key = canon(P[a], P[b], P[c])
        v = seen.get(key)
        if v is None:
            v = p2(*key); seen[key] = v
        if not v:
            if nz[a, b, c]: fz += 1
            else: caught += 1
    tot = int(hard.sum()); acc = 1 - (tot - caught + fz) / nz.size
    print(f"n={n}: hard zeros {tot}, caught by P2 {caught} ({caught / max(1, tot):.3f}), false zeros {fz}, accuracy {acc:.4f}, {time.time() - t0:.0f}s", flush=True)
