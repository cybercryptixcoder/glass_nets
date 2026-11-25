import sys, numpy as np
sys.path.insert(0, "scripts/kronecker")
from kron_feats import load, b1_masks, conj
def s(p): return "".join(str(x) for x in p if x) if max(p) < 10 else ",".join(str(x) for x in p if x)
for n in map(int, sys.argv[1:]):
    parts, g = load(n); nz = g != 0; ok = b1_masks(parts); hard = ok & ~nz
    P = len(parts); C = conj(parts); key = {tuple(p): i for i, p in enumerate(parts.tolist())}; cj = [key[tuple(c)] for c in C.tolist()]
    seen = set(); reps = []
    for a, b, c in zip(*np.nonzero(hard)):
        forms = []
        for (x, y, z) in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
            for (u, v, w) in [(x,y,z),(cj[x],cj[y],z),(cj[x],y,cj[z]),(x,cj[y],cj[z])]:
                forms.append((u, v, w))
        f = min(forms)
        if f not in seen:
            seen.add(f); reps.append(f)
    print(f"n={n}: {hard.sum()} hard zeros, {len(reps)} orbits")
    for f in reps[:60]:
        print("  ", " | ".join(s(parts[i]) for i in f))
