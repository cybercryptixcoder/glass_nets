"""Known-math lines P3r (KR1c) and P4 (KR1d) on held-out n, aggregate numbers only (accuracy, false zeros,
missed zeros); same orbit evaluation as p2_heldout.py. Usage: p34_heldout.py MODE n  (MODE = p3r or p4).
Writes results/kronecker/heldout_<MODE>_n<n>.json."""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import face_reduce as fr
from gwxz import gwxz_zero
from fathook import fathook_zero
mode, n = sys.argv[1], int(sys.argv[2])
_b1 = fr.b1
if mode == "p3r":
    fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t)
elif mode == "p4":
    fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t) and not fathook_zero(A, B, t)
else:
    raise SystemExit("mode must be p3r or p4")
fr.p2.cache_clear()
from kron_feats import load, b1_masks
from b2_lgbm_lib import orbit_ids
t0 = time.time()
parts, g = load(n); P = len(parts); Pp = [tuple(int(x) for x in p if x) for p in parts]
nz = (g != 0).ravel(); ok = b1_masks(parts).ravel()
orb = orbit_ids(parts).ravel(); del g
uo, inv = np.unique(orb, return_inverse=True); del orb
u, rem = np.divmod(uo, P * P); v, w = np.divmod(rem, P)
pred_u = np.zeros(len(uo), dtype=bool)
for i in np.flatnonzero(ok[uo]):
    pred_u[i] = fr.p2(Pp[u[i]], Pp[v[i]], Pp[w[i]])
pred = pred_u[inv]
res = dict(n=n, mode=mode, triples=int(nz.size), orbits=int(len(uo)), acc=float((pred == nz).mean()),
           false_zero=int((~pred & nz).sum()), missed_zeros=int((pred & ~nz).sum()), seconds=round(time.time() - t0))
print(res, flush=True)
os.makedirs("results/kronecker", exist_ok=True)
json.dump(res, open(f"results/kronecker/heldout_{mode}_n{n}.json", "w"), indent=1)
