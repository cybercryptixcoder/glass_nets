"""KR1g (descriptive): re-train the KR1b (A) LightGBM exactly as p2_heldout.py does and cross-tabulate its test
errors against P2 and P4 (does known math get each of the black box's mistakes right?).
Usage: lgbm_errors.py n   -> results/kronecker/lgbm_errors_n<n>.json"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_reduce as fr
from gwxz import gwxz_zero
from fathook import fathook_zero
from kron_feats import load, b1_masks
from b2_lgbm_lib import orbit_ids
n = int(sys.argv[1]); t0 = time.time()
parts, g = load(n); P = len(parts); Pp = [tuple(int(x) for x in p if x) for p in parts]
nz = (g != 0).ravel(); ok = b1_masks(parts).ravel(); orb = orbit_ids(parts).ravel(); del g
uo, inv = np.unique(orb, return_inverse=True); inv = inv.ravel()
u, rem = np.divmod(uo, P * P); v, w = np.divmod(rem, P)
def line(patch):
    _b1 = fr.b1
    if patch: fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t) and not fathook_zero(A, B, t)
    fr.p2.cache_clear(); out = np.zeros(len(uo), dtype=bool)
    for i in np.flatnonzero(ok[uo]): out[i] = fr.p2(Pp[u[i]], Pp[v[i]], Pp[w[i]])
    fr.b1 = _b1; fr.p2.cache_clear(); return out[inv]
p2 = line(False); p4 = line(True)
import lightgbm as lgb
rng = np.random.default_rng(0)
test_orbits = rng.choice(uo, size=len(uo) // 5, replace=False)
te = np.isin(uo, test_orbits)[inv]
tr_idx = np.flatnonzero(~te); te_idx = np.flatnonzero(te)
if len(tr_idx) > 4_000_000: tr_idx = rng.choice(tr_idx, 4_000_000, replace=False)
if len(te_idx) > 2_000_000: te_idx = rng.choice(te_idx, 2_000_000, replace=False)
def feats(idx):
    a, r = np.divmod(idx, P * P); b, c = np.divmod(r, P)
    return np.concatenate([parts[a], parts[b], parts[c]], 1).astype(np.float32)
m = lgb.LGBMClassifier(n_estimators=600, num_leaves=255, learning_rate=0.1, verbose=-1, n_jobs=int(os.environ.get("NJOBS", "2")))
m.fit(feats(tr_idx), nz[tr_idx]); lg = m.predict(feats(te_idx)).astype(bool)
truth = nz[te_idx]; acc = float((lg == truth).mean())
rec = json.load(open(f"results/kronecker/heldout_n{n}.json"))["lgbm_acc_test"]
err = lg != truth
res = dict(n=n, lgbm_acc_test=acc, recorded_lgbm_acc_test=rec, reproduces=abs(acc - rec) < 5e-5, lgbm_errors=int(err.sum()),
           lgbm_false_zeros=int((~lg & truth).sum()), lgbm_missed_zeros=int((lg & ~truth).sum()),
           p2_right_on_lgbm_errors=float((p2[te_idx][err] == truth[err]).mean()), p4_right_on_lgbm_errors=float((p4[te_idx][err] == truth[err]).mean()),
           p2_errors=int((p2[te_idx] != truth).sum()), lgbm_right_on_p2_errors=float((lg[p2[te_idx] != truth] == truth[p2[te_idx] != truth]).mean()),
           seconds=round(time.time() - t0))
print(res, flush=True)
json.dump(res, open(f"results/kronecker/lgbm_errors_n{n}.json", "w"), indent=1)
