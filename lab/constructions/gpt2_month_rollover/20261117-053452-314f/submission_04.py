import numpy as np

DESCRIPTION = """Third pass on cut 12 (stage layers 9-11). What the end-position residual at depth 12
must deliver to ln_f + W_U is the identity of the prompt's year (it picks which of the 20 year
token triples the answer logits are read from) plus the two answer margins T = logit(next)-logit(same)
and U = logit(prev)-logit(same). Measured: logits = ln_f(resid12@end) @ W_U exactly; T is readable
from resid12@end with R2 0.9999 but only by a *per-year* set of directions (the graded direction
W_U[year+1]-W_U[year] changes with the year), so the year identity enters the cut as a one-hot.
Behaviour: 'previous year' is never chosen (U = -1.52 +- 0.5, always negative); the decision is the
sign of T, which is set by m2 (monotone), m1 (weak) and a 12x12 m1 x m2 interaction (T is high when
m2 = m1, low when m2 comes right after m1) plus a weak jump-year ripple in the year."""

A1 = np.array([-0.30,-0.11,-0.06,-0.31, 0.17,-0.05,-0.07, 0.42, 0.37,-0.11, 0.12,-0.08])
A2 = np.array([ 1.55, 1.09, 0.62, 0.55,-0.06, 0.22, 0.03,-0.30,-0.61,-0.87,-1.08,-1.12])
EY = np.array([-0.25,-0.09, 0.12, 0.26,-0.13, 0.05, 0.24, 0.45,-0.32,-0.14, 0.16, 0.24,
               -0.33,-0.13, 0.12, 0.28,-0.34,-0.24,-0.05, 0.08])
R = np.array([
 [ 0.36,-1.17,-0.46,-0.42, 0.25,-0.02, 0.14, 0.06, 0.27, 0.30, 0.34, 0.33],
 [-0.10, 0.66,-1.13,-0.73,-0.24, 0.02, 0.18, 0.14, 0.05, 0.33, 0.41, 0.41],
 [-0.12,-0.04, 0.38,-1.04,-0.26,-0.38, 0.14, 0.12, 0.10, 0.26, 0.39, 0.45],
 [-0.11, 0.08, 0.06, 0.80,-0.64,-0.43,-0.21, 0.03, 0.05, 0.18, 0.02, 0.18],
 [ 0.02, 0.00, 0.18, 0.02, 0.57,-0.77,-0.28,-0.17, 0.05, 0.07, 0.16, 0.15],
 [ 0.10, 0.26, 0.29, 0.29, 0.29, 0.71,-0.75,-0.62,-0.55, 0.04,-0.05,-0.00],
 [ 0.11, 0.20, 0.37, 0.26, 0.35, 0.28, 0.65,-0.88,-0.63,-0.44,-0.06,-0.20],
 [ 0.07, 0.23, 0.27, 0.30, 0.02, 0.22, 0.11, 0.64,-0.91,-0.39,-0.41,-0.14],
 [ 0.04,-0.00, 0.16, 0.23, 0.12, 0.11, 0.07, 0.16, 0.95,-0.87,-0.60,-0.37],
 [ 0.03, 0.03,-0.04, 0.18,-0.17, 0.14,-0.00, 0.27, 0.28, 0.82,-1.10,-0.44],
 [-0.00,-0.13, 0.01, 0.13,-0.05, 0.05, 0.02, 0.15, 0.22,-0.16, 0.98,-1.21],
 [-0.39,-0.13,-0.10,-0.01,-0.24, 0.07,-0.08, 0.10, 0.14,-0.13,-0.09, 0.85]])
C0 = 0.879 - (A1.mean() + A2.mean() + EY.mean())

def model(x, iv):
    m1 = x[:, 0].astype(int); m2 = x[:, 1].astype(int); yi = x[:, 2].astype(int) - 1980
    T = C0 + A1[m1] + A2[m2] + EY[yi] + R[m1, m2]
    U = np.full_like(T, -1.52)
    Yhat = iv("year_identity", np.eye(20)[yi])
    T = iv("T_margin", T)
    U = iv("U_margin", U)
    cs = np.full_like(T, -149.03)
    return np.stack([cs, cs + T, cs + U], axis=1)

ALIGN = {"year_identity": ["resid12@end"], "T_margin": ["resid12@end"], "U_margin": ["resid12@end"]}
