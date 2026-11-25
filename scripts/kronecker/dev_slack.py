import sys, numpy as np
sys.path.insert(0, "scripts/kronecker")
from kron_feats import load, b1_masks, conj, inter
for n in (12, 14):
    parts, g = load(n); nz = g != 0; ok = b1_masks(parts); hard = ok & ~nz
    C = conj(parts); r1, ln = parts[:, 0], C[:, 0]
    I, Ic, Imc = inter(parts, parts), inter(C, C), inter(parts, C)
    s_row = np.minimum(I, Ic)[:, :, None] - r1[None, None, :]          # slack of the row bound, target slot 3
    s_col = np.minimum(Imc, Imc.T)[:, :, None] - ln[None, None, :]     # slack of the column bound, target slot 3
    s = np.minimum(s_row, s_col)
    smin = np.minimum(np.minimum(s, s.transpose(2, 0, 1)), s.transpose(1, 2, 0))
    print(f"n={n}")
    for k in range(0, 5):
        sel = ok & (smin == k)
        print(f"   min Dvir slack={k}: B1-pass {int(sel.sum()):8d} hard zeros {int((hard & sel).sum()):6d} rate {(hard & sel).sum() / max(1, sel.sum()):.3f}")
    sel = ok & (smin >= 5); print(f"   slack>=5: B1-pass {int(sel.sum()):8d} hard zeros {int((hard & sel).sum()):6d} rate {(hard & sel).sum() / max(1, sel.sum()):.3f}")
