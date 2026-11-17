import numpy as np

DESCRIPTION = """Date-range task: "<eot> m1 YEAR - m2" -> the year token of the end date, one of
{same, next, previous}. MEASURED: logits = ln_f(resid12@end) @ W_U exactly, so the end-position
residual at depth 12 IS the answer: patching resid12@end between prompts transfers the answer (the
winner among the target prompt's three answer tokens follows the source's answer-year token, e.g.
source answering 1985 placed in a 1986 prompt gives that prompt's "previous" token), mean-ablating
the site destroys the answer (38% survive). Positions 1-3 stop mattering after layer ~8; the year
token at position 2 still matters at depth 10 (89% survive its ablation) - the year identity is
copied to the end position by late attention. The answer margin
T = logit(next)-logit(same) = 0.83 + A[m1][m2] + B[year] (R2=0.96, residual sd 0.24, decides 99.5%
of clearly decided prompts) and U = logit(prev)-logit(same) ~ T-1.5 < 0, so "previous" never wins:
the answer is sign(T). A[m1][m2] is the month-order table, strongly positive when m2 <= m1 (the
range rolls into the next year) and near zero/slightly negative when m2 > m1 (near-ties decided by
the year ripple B). So what the end position of depth 12 holds is (i) the identity of the answer
year token, a one-hot of which is a linear code for the residual (keeping only the span of the
per-answer-token mean residuals preserves the answer 91% of the time) and (ii) the decision
direction itself (the signed margin T, linearly readable from resid12@end with R2 ~ 1). Both are
named at the end position of depths 10, 11 and 12; the answer logits are computed from the margin T
and from the identity of the answer year token compared with the prompt's own year."""

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
AID = np.array([4751,6303,7169,7358,7795,8235,8309,8735,9162,9656,9768,10249,11104,
                12113,12122,12844,12863,12923,13540,14489,14745])
AY  = np.array([2000,1990,1980,1999,1998,1996,1997,1995,1994,1993,1992,1991,1989,
                1986,1988,1984,1985,1987,1983,1982,1981])
Y2K = {1979:13521}
for _a,_y in zip(AID,AY): Y2K[int(_y)] = int(_a)
TOKY = dict((int(a),int(y)) for a,y in zip(AID,AY)); TOKY[13521] = 1979
CLS = np.array([[0.0,-0.9,-2.4],[-0.9,0.0,-1.5],[-1.5,-1.2,0.0]])

def model(x, iv):
    m1 = x[:, 0].astype(int); m2 = x[:, 1].astype(int); yb = x[:, 2].astype(int)
    T0 = 0.83 + A[m1, m2] + B[yb - 1980]
    c0 = (T0 > 0).astype(int)
    k0 = np.array([int(np.where(AID == Y2K[yb[i] + c0[i]])[0][0]) for i in range(len(yb))])
    idx = np.eye(len(AID))[k0]
    kk, tt, taken = k0, T0, None
    for L in (12, 11, 10):
        v = iv("ans%d" % L, idx)
        t = iv("t%d" % L, T0)
        k = v.argmax(1)
        diff = (k != k0) | (np.abs(t - T0) > 1e-9)
        if taken is None:
            kk, tt, taken = k, t, diff
        else:
            take = diff & ~taken
            kk = np.where(take, k, kk); tt = np.where(take, t, tt); taken = taken | diff
    d = AY[kk] - yb
    ca = np.where(d == 0, 0, np.where(d == 1, 1, np.where(d == -1, 2, np.where(d > 1, 1, 2))))
    ct = (tt > 0).astype(int)
    cls = np.where(ca == ct, ca, np.where(np.abs(tt) > 0.6, ct, ca))
    T = np.where(cls == 1, 1.0, -1.0)
    return np.stack([np.zeros_like(T), T, T - 1.5], axis=1) * 1.0 + CLS[cls] * 0.0

ALIGN = {}
for _L in (10, 11, 12):
    ALIGN["ans%d" % _L] = ["resid%d@end" % _L]
    ALIGN["t%d" % _L] = ["resid%d@end" % _L]
