"""Kill test, part 2: LightGBM on raw parts (B2), per n, split 80/20 by symmetry orbit; scored overall and on
the triples B1 cannot decide (where all hard zeros live)."""
import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kron_feats import load, b1_masks, conj

def orbit_ids(parts):
    """Canonical id of each triple under permutations of the three arguments and conjugation of an even number."""
    P = len(parts); C = conj(parts)
    key = {tuple(p): i for i, p in enumerate(parts.tolist())}
    cj = np.array([key[tuple(c)] for c in C.tolist()])
    a, b, c = np.meshgrid(np.arange(P), np.arange(P), np.arange(P), indexing="ij")
    best = None
    for (x, y, z) in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
        for (u, v, w) in [(x, y, z), (cj[x], cj[y], z), (cj[x], y, cj[z]), (x, cj[y], cj[z])]:
            idx = (u * P + v) * P + w
            best = idx if best is None else np.minimum(best, idx)
    return best

res = {}
for n in map(int, sys.argv[1:]):
    parts, g = load(n); P = len(parts); nz = (g != 0).ravel(); ok = b1_masks(parts).ravel()
    orb = orbit_ids(parts).ravel()
    rng = np.random.default_rng(0); uo = np.unique(orb); test_orbits = set(rng.choice(uo, size=len(uo) // 5, replace=False).tolist())
    te = np.isin(orb, np.array(sorted(test_orbits)))
    a, b, c = np.meshgrid(np.arange(P), np.arange(P), np.arange(P), indexing="ij")
    X = np.concatenate([parts[a.ravel()], parts[b.ravel()], parts[c.ravel()]], 1).astype(np.float32)
    tr = ~te
    if tr.sum() > 4_000_000:
        sub = np.flatnonzero(tr); tr = np.zeros_like(tr); tr[rng.choice(sub, 4_000_000, replace=False)] = True
    m = lgb.LGBMClassifier(n_estimators=600, num_leaves=255, learning_rate=0.1, verbose=-1, n_jobs=2)
    m.fit(X[tr], nz[tr])
    pr = m.predict(X[te]).astype(bool)
    acc = float((pr == nz[te]).mean()); b1 = float((ok[te] == nz[te]).mean())
    hz = ok[te] & ~nz[te]
    caught = float((~pr[hz]).mean()) if hz.any() else float("nan")
    fneg = float((~pr[nz[te]]).mean())
    res[n] = dict(test_triples=int(te.sum()), lgbm_acc=acc, b1_acc=b1, hard_zeros_test=int(hz.sum()),
                  lgbm_catches_hard_zero_frac=caught, lgbm_false_zero_rate_on_nonzero=fneg)
    print(n, res[n], flush=True)
json.dump(res, open(f"results/kronecker/b2_lgbm_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
