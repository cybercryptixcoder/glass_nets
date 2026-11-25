"""Dev: does the black box (LightGBM, same orbit split as b2_lgbm.py) know anything P2 doesn't?"""
import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kron_feats import load, b1_masks
from b2_lgbm_lib import orbit_ids
from face_reduce import p2, canon
out = {}
for n in map(int, sys.argv[1:]):
    parts, g = load(n); Pp = [tuple(int(x) for x in p if x) for p in parts]; P = len(parts)
    nz = (g != 0).ravel(); ok = b1_masks(parts).ravel(); orb = orbit_ids(parts).ravel()
    rng = np.random.default_rng(0); uo = np.unique(orb); test_orbits = rng.choice(uo, size=len(uo) // 5, replace=False)
    te = np.isin(orb, test_orbits); tr = ~te
    a, b, c = np.meshgrid(np.arange(P), np.arange(P), np.arange(P), indexing="ij")
    A, B, C = a.ravel(), b.ravel(), c.ravel()
    X = np.concatenate([parts[A], parts[B], parts[C]], 1).astype(np.float32)
    m = lgb.LGBMClassifier(n_estimators=600, num_leaves=255, learning_rate=0.1, verbose=-1, n_jobs=2).fit(X[tr], nz[tr])
    lg = m.predict(X[te]).astype(bool)
    ti = np.flatnonzero(te)
    p2p = np.array([ok[i] and p2(*canon(Pp[A[i]], Pp[B[i]], Pp[C[i]])) for i in ti])
    truth = nz[te]
    resid = p2p & ~truth                                   # zeros P2 misses
    r = dict(test=int(te.sum()), p2_acc=float((p2p == truth).mean()), lgbm_acc=float((lg == truth).mean()),
             p2_false_zero=int((~p2p & truth).sum()), residue=int(resid.sum()),
             lgbm_catches_residue=float((~lg[resid]).mean()) if resid.any() else None,
             lgbm_false_zero=int((~lg & truth).sum()),
             combo_acc=float(((p2p & lg) == truth).mean()))
    out[n] = r; print(n, r, flush=True)
json.dump(out, open(f"results/kronecker/p2_vs_lgbm_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
