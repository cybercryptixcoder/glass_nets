import numpy as np

DESCRIPTION = """Stage layers 9-11, cut at resid12@end. The picture, measured:
- logits at the end position = ln_f(resid12@end) @ W_U exactly; patching resid12@end from a prompt
  answered 'same year' into a prompt answered 'next year' transfers that prompt's margins exactly
  (T,U match to 0.01) and flips the answer 24/24. The end residual *is* the answer.
- The decision is the sign of T = logit(next)-logit(same). 'previous year' never wins (U = -1.5 +- 0.5,
  always below T).
- T = Tbar + A[m1,m2] + B[year], R2 0.96, residual sd 0.24. A is a clean month-order table: T is
  positive when the end month is at or before the start month (the range rolls into the next year) and
  negative once m2 passes m1; the transition band is a diagonal ~1.3 wide. B is a small (~0.6) year
  effect. Fitted on all 2880 prompts, this decides 99.9% of the clearly-decided prompts.
- The state at depth 12 is high dimensional (hundreds of PCs are needed to keep the answers), and its
  variance is dominated by the year-token identity. A pairwise tensor code of the three input
  identities -- the month pair (m1,m2), (m2,year) and (m1,year) -- reproduces 97.5% of the residual's
  variance and gives back the margins with the right amplitude, so those identities are the variables
  I place at this cut (all categorical, each value occurs in 12 or more prompts)."""

A = np.array([
 [ 1.21,-0.78,-0.54,-0.57,-0.51,-0.50,-0.53,-0.94,-1.04,-1.27,-1.44,-1.49],
 [ 0.59, 0.89,-1.37,-1.04,-1.16,-0.62,-0.65,-1.02,-1.42,-1.40,-1.53,-1.57],
 [ 0.61, 0.24, 0.18,-1.30,-1.14,-0.98,-0.65,-0.99,-1.33,-1.42,-1.50,-1.49],
 [ 0.82, 0.56, 0.06, 0.73,-1.32,-0.82,-0.79,-0.88,-1.18,-1.30,-1.68,-1.56],
 [ 1.02, 0.54, 0.25, 0.02,-0.04,-1.10,-0.80,-1.02,-1.11,-1.35,-1.47,-1.52],
 [ 1.41, 1.12, 0.68, 0.61,-0.00, 0.70,-0.95,-1.15,-1.39,-1.06,-1.36,-1.36],
 [ 1.71, 1.34, 1.04, 0.86, 0.34, 0.55, 0.73,-1.12,-1.19,-1.25,-1.08,-1.27],
 [ 1.84, 1.54, 1.11, 1.08, 0.18, 0.66, 0.36, 0.57,-1.30,-1.03,-1.27,-1.04],
 [ 2.03, 1.53, 1.22, 1.22, 0.50, 0.77, 0.55, 0.30, 0.78,-1.29,-1.23,-1.05],
 [ 2.35, 1.89, 1.35, 1.50, 0.54, 1.13, 0.80, 0.74, 0.44, 0.72,-1.41,-0.79],
 [ 2.67, 2.09, 1.76, 1.81, 1.02, 1.40, 1.18, 0.98, 0.74, 0.10, 1.03,-1.20],
 [ 2.36, 2.16, 1.72, 1.74, 0.90, 1.49, 1.15, 1.00, 0.73, 0.21, 0.03, 0.93]])
B = np.array([0.11,-0.54,0.15,0.13,-0.43,-0.15,0.05,0.22,0.18,-0.39,0.35,-0.23,
              0.69,0.65,-0.05,-0.03,0.39,0.07,-0.12,-0.23])
T0 = 0.8318

def model(x, iv):
    m1 = x[:, 0].astype(int); m2 = x[:, 1].astype(int); yi = x[:, 2].astype(int) - 1980
    mp = iv("month_pair", np.eye(144)[m1 * 12 + m2])
    my = iv("m2_year", np.eye(240)[m2 * 20 + yi])
    m1y = iv("m1_year", np.eye(240)[m1 * 20 + yi])
    T = T0 + A[m1, m2] + B[yi]
    T = iv("T_margin", T)
    cs = np.full_like(T, -149.03)
    return np.stack([cs, cs + T, cs + T - 1.5], axis=1)

ALIGN = {"month_pair": ["resid12@end"], "m2_year": ["resid12@end"],
         "m1_year": ["resid12@end"], "T_margin": ["resid12@end"]}
