import numpy as np

DESCRIPTION = """Date-range task: "<eot> m1 YEAR - m2"; answer = year token of the end date, one of
{same, next, previous}. MEASURED: (a) logits = ln_f(resid12@end) @ W_U exactly, so the end-position
residual at depth 12 IS the answer; patching resid12@end between prompts transfers the answer
exactly (24/24 flips), mean-ablating it destroys it (36% decisions survive). (b) Positions 1-3 stop
mattering after layer ~8; at depth 10 the year token at position 2 still matters (89% survive its
ablation) — the year identity is carried by late attention to the end position. (c) The answer
margin T = logit(next)-logit(same) is a two-way function of the two months and the year,
T = 0.83 + A[m1][m2] + B[year] (fit R2=0.96, residual sd 0.24, decides 99.5% of clearly decided
prompts), and U = logit(prev)-logit(same) ~ T - 1.5 < 0, so "previous" never wins. A[m1][m2] is the
month-order table: strongly POSITIVE (1-3) when m2 <= m1 (range rolls into the next year), near zero
slightly negative when m2 > m1 (those prompts are decided by the small year ripple B, hence about a
quarter of them still answer "next"). So the mechanism is: the end position holds the year identity
(copied from position 2) together with the m1/m2 comparison, and the three answer logits are just
[0, T, T-1.5] with T built from an (m1 x m2) table plus a year-dependent offset. The identity of
(m1,m2) and of the year are placed as one compact categorical code at the end position of depths 10,
11 and 12, where that information sits."""

A = np.array([
 [ 1.21,-0.78,-0.54,-0.57,-0.51,-0.50,-0.53,-0.94,-1.04,-1.27,-1.44,-1.49],
 [ 0.59, 0.89,-1.37,-1.04,-1.16,-0.62,-0.65,-1.02,-1.40,-1.40,-1.53,-1.57],
 [ 0.61, 0.24, 0.18,-1.30,-1.14,-0.98,-0.65,-0.99,-1.33,-1.42,-1.50,-1.49],
 [ 0.82, 0.56, 0.06, 0.73,-1.32,-0.82,-0.79,-0.88,-1.18,-1.30,-1.68,-1.56],
 [ 1.02, 0.54, 0.25, 0.02,-0.04,-1.10,-0.80,-1.02,-1.11,-1.35,-1.47,-1.52],
 [ 1.41, 1.12, 0.68, 0.61, 0.00, 0.70,-0.95,-1.15,-1.39,-1.06,-1.36,-1.36],
 [ 1.71, 1.34, 1.04, 0.86, 0.34, 0.55, 0.73,-1.12,-1.19,-1.25,-1.08,-1.27],
 [ 1.84, 1.54, 1.11, 1.08, 0.18, 0.66, 0.36, 0.57,-1.30,-1.03,-1.27,-1.04],
 [ 2.03, 1.53, 1.22, 1.22, 0.50, 0.77, 0.55, 0.30, 0.78,-1.29,-1.23,-1.05],
 [ 2.35, 1.89, 1.35, 1.50, 0.54, 1.13, 0.80, 0.74, 0.44, 0.72,-1.41,-0.79],
 [ 2.67, 2.09, 1.76, 1.81, 1.02, 1.40, 1.18, 0.98, 0.74, 0.10, 1.03,-1.20],
 [ 2.36, 2.16, 1.72, 1.74, 0.90, 1.49, 1.15, 1.00, 0.73, 0.21, 0.03, 0.93]])
B = np.array([0.11,-0.54,0.15,0.13,-0.43,-0.15,0.05,0.22,0.18,-0.39,
              0.35,-0.23,0.69,0.65,-0.05,-0.03,0.39,0.07,-0.12,-0.23])

def model(x, iv):
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    yy = x[:, 2].astype(int) - 1980
    code = np.concatenate([np.eye(20)[yy], np.eye(144)[m1 * 12 + m2]], axis=1)
    M1, M2, YR = m1, m2, yy
    taken = None
    for L in (12, 11, 10):
        v = iv("code%d" % L, code)
        c1 = v[:, :20].argmax(1)
        ij = v[:, 20:].argmax(1)
        d1, d2 = ij // 12, ij % 12
        diff = (d1 != m1) | (d2 != m2) | (c1 != yy)
        if taken is None:
            M1, M2, YR, taken = d1, d2, c1, diff
        else:
            take = diff & ~taken
            M1 = np.where(take, d1, M1)
            M2 = np.where(take, d2, M2)
            YR = np.where(take, c1, YR)
            taken = taken | diff
    T = 0.83 + A[M1, M2] + B[YR]
    U = T - 1.5
    return np.stack([np.zeros_like(T), T, U], axis=1)

ALIGN = {"code10": ["resid10@end"], "code11": ["resid11@end"], "code12": ["resid12@end"]}
