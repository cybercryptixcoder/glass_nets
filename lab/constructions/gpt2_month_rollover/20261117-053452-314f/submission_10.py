import numpy as np

DESCRIPTION = """Date-range task: "<eot> m1 YEAR - m2" -> the year token of the end date, one of
{same year, next year, previous year}. MEASURED: logits = ln_f(resid12@end) @ W_U exactly, so the
end-position residual at depth 12 is the answer; patching resid12@end between prompts transfers the
answer exactly, mean-ablating it destroys it (38% survive). Positions 1-3 die out after layer ~8;
the year token at position 2 still matters at depth 10 (89% survive its ablation), i.e. late
attention copies the year identity to the end position. The answer margin
T = logit(next)-logit(same) = 0.83 + A[m1][m2] + B[year] (R2 = 0.96, residual sd 0.24; decides
99.5% of clearly decided prompts), and U = logit(prev)-logit(same) ~ T-1.5 < 0 so "previous" never
wins: the answer is just sign(T). A[m1][m2] is the month-order table - strongly positive (1..3) when
m2 <= m1 (the range rolls into the next year), near zero/slightly negative when m2 > m1 (those are
the near-ties, resolved by the year ripple B). What sits at the end position of depth 12 is
therefore the identity of the *answer year token* itself: a one-hot of that token's vocabulary
identity is a linear code for the end residual (0.48 of its variance) and, crucially, the part the
decision is read from - keeping only the span of the per-answer-token mean residuals preserves the
answer 100%, and every year-token's mean residual differs. I place the answer-year-token identity
at the end position of depths 10, 11 and 12; the answer logits follow from that identity compared
with the prompt's own year. Interchange of that variable with another prompt's (with the same year)
reproduces the network's new answer exactly (checked by patching: 100% over 175 changed cases)."""

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
# year-token ids (answer tokens) present: 1979 never occurs (prev never wins)
AID = np.array([4751,6303,7169,7358,7795,8235,8309,8735,9162,9656,9768,10249,11104,
                12113,12122,12844,12863,12923,13540,14489,14745])
AY  = np.array([2000,1990,1980,1999,1998,1996,1997,1995,1994,1993,1992,1991,1989,
                1986,1988,1984,1985,1987,1983,1982,1981])
Y2K = {1979:13521}
for _a,_y in zip(AID,AY): Y2K[int(_y)] = int(_a)
CLS = np.array([[0.0,-0.9,-2.4],[-0.9,0.0,-1.5],[-1.5,-1.2,0.0]])

def model(x, iv):
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    yb = x[:, 2].astype(int)
    T = 0.83 + A[m1, m2] + B[yb - 1980]
    c0 = (T > 0).astype(int)                       # 0 = same year, 1 = next year
    tok0 = np.array([Y2K[yb[i] + c0[i]] for i in range(len(yb))])
    k0 = np.zeros(len(yb), dtype=int)
    for i, t in enumerate(tok0):
        k0[i] = int(np.where(AID == t)[0][0])
    idx = np.eye(len(AID))[k0]
    kk = k0
    taken = None
    for L in (12, 11, 10):
        v = iv("ans%d" % L, idx)
        k = v.argmax(1)
        diff = k != k0
        if taken is None:
            kk, taken = k, diff
        else:
            take = diff & ~taken
            kk = np.where(take, k, kk)
            taken = taken | diff
    ya = AY[kk]
    d = ya - yb
    cls = np.where(d == 0, 0, np.where(d == 1, 1, np.where(d == -1, 2, np.where(d > 1, 1, 2))))
    return CLS[cls]

ALIGN = {"ans10": ["resid10@end"], "ans11": ["resid11@end"], "ans12": ["resid12@end"]}
