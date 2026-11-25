import sys, numpy as np
sys.path.insert(0, "scripts/kronecker")
from kron_feats import load, b1_masks, conj
for n in (10, 12, 14):
    parts, g = load(n); nz = g != 0; ok = b1_masks(parts); hard = ok & ~nz
    C = conj(parts); d = np.minimum(n - parts[:, 0], n - C[:, 0])      # distance to trivial or sign
    P = len(parts)
    dmin = np.minimum(np.minimum(d[:, None, None], d[None, :, None]), d[None, None, :])
    print(f"n={n}: hard zeros {hard.sum()}, B1-pass {ok.sum()}")
    for k in range(0, 6):
        sel = dmin == k
        print(f"   d_min={k}: B1-pass {int((ok & sel).sum()):9d}  hard zeros {int((hard & sel).sum()):7d}  rate {((hard & sel).sum() / max(1, (ok & sel).sum())):.3f}")
    sel = dmin >= 6
    print(f"   d_min>=6: B1-pass {int((ok & sel).sum()):9d}  hard zeros {int((hard & sel).sum()):7d}  rate {((hard & sel).sum() / max(1, (ok & sel).sum())):.3f}")
    # same partition repeated?
    eq = (np.arange(P)[:, None, None] == np.arange(P)[None, :, None]) | (np.arange(P)[None, :, None] == np.arange(P)[None, None, :]) | (np.arange(P)[:, None, None] == np.arange(P)[None, None, :])
    print(f"   with two equal arguments: hard zeros {int((hard & eq).sum())}")
