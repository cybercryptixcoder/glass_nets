"""KR1b (A) on held-out n: P2 (frozen, scripts/kronecker/face_reduce.py) against exact truth on every ordered
triple, and LightGBM (B2) trained per n on an 80% orbit split, both scored on the same test triples.
Usage: python3 p2_heldout.py n  (tables in $KDATA). Writes results/kronecker/heldout_n<n>.json."""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kron_feats import load, b1_masks
from b2_lgbm_lib import orbit_ids
from face_reduce import p2
n = int(sys.argv[1]); t0 = time.time()
parts, g = load(n); P = len(parts); Pp = [tuple(int(x) for x in p if x) for p in parts]
nz = (g != 0).ravel(); ok = b1_masks(parts).ravel()
orb = orbit_ids(parts).ravel()
uo, inv = np.unique(orb, return_inverse=True)
u, rem = np.divmod(uo, P * P); v, w = np.divmod(rem, P)
okr = ok[uo]                                    # B1 is symmetric, so the representative's value is the orbit's
pred_u = np.zeros(len(uo), dtype=bool)
for i in np.flatnonzero(okr):
    pred_u[i] = p2(Pp[u[i]], Pp[v[i]], Pp[w[i]])
pred = pred_u[inv]
t1 = time.time()
res = dict(n=n, triples=int(nz.size), orbits=int(len(uo)), nonzero=float(nz.mean()),
           table_sha1=hashlib.sha1(np.ascontiguousarray(g).tobytes()).hexdigest(),
           b1_acc=float((ok == nz).mean()), p2_acc=float((pred == nz).mean()),
           p2_false_zero=int((~pred & nz).sum()), p2_missed_zeros=int((pred & ~nz).sum()),
           hard_zeros=int((ok & ~nz).sum()), p2_seconds=round(t1 - t0))
print(res, flush=True)
# LightGBM on an orbit split (seed 0), trained on up to 4M training triples, scored on up to 2M test triples
import lightgbm as lgb
rng = np.random.default_rng(0)
test_orbits = rng.choice(uo, size=len(uo) // 5, replace=False)
te_mask_u = np.isin(uo, test_orbits); te = te_mask_u[inv]
tr_idx = np.flatnonzero(~te); te_idx = np.flatnonzero(te)
if len(tr_idx) > 4_000_000: tr_idx = rng.choice(tr_idx, 4_000_000, replace=False)
if len(te_idx) > 2_000_000: te_idx = rng.choice(te_idx, 2_000_000, replace=False)
def feats(idx):
    a, r = np.divmod(idx, P * P); b, c = np.divmod(r, P)
    return np.concatenate([parts[a], parts[b], parts[c]], 1).astype(np.float32)
m = lgb.LGBMClassifier(n_estimators=600, num_leaves=255, learning_rate=0.1, verbose=-1, n_jobs=int(os.environ.get("NJOBS", "2")))
m.fit(feats(tr_idx), nz[tr_idx])
lg = m.predict(feats(te_idx)).astype(bool)
truth = nz[te_idx]; p2t = pred[te_idx]
res.update(test_triples=int(len(te_idx)), lgbm_acc_test=float((lg == truth).mean()), p2_acc_test=float((p2t == truth).mean()),
           b1_acc_test=float((ok[te_idx] == truth).mean()), lgbm_false_zero_test=int((~lg & truth).sum()),
           lgbm_catches_p2_residue=float((~lg[p2t & ~truth]).mean()) if (p2t & ~truth).any() else None,
           total_seconds=round(time.time() - t0))
print(res, flush=True)
os.makedirs("results/kronecker", exist_ok=True)
json.dump(res, open(f"results/kronecker/heldout_n{n}.json", "w"), indent=1)
