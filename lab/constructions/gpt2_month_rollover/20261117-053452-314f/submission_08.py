import numpy as np

DESCRIPTION = """Date-range task: "<eot> m1 YEAR - m2", answer = the year token for the end date,
one of {same year, next year, previous year}. Measured facts: logits = ln_f(resid12@end) @ W_U
exactly, so the end-position residual at depth 12 IS the answer; patching resid12@end between
prompts transfers the answer exactly, while mean-ablating it destroys it. Positions 1-3 stop
mattering after ~layer 8 (only the end position carries the answer late); at depth 10 the year
token at position 2 still matters (0.89 survival on ablation). The answer margin
T = logit(next)-logit(same) is a function of BOTH months (a month-order term) and the year (a
small ripple), i.e. it has genuine (m1 x m2) and (m x year) product structure, so single
additive one-hot codes explain only ~0.68 of the residual; pairwise identity one-hots
(m1,m2), (m1,year), (m2,year) are needed. Margins measured on examples: "January 1980 -
February" T=+0.42, "January 1980 - March" T=+0.62, "January 1980 - January" T=+2.31, i.e. the
model picks NEXT year when m2 comes after m1, and is near-tied (decided by the year ripple, so
about half next) when m2 <= m1. logit(prev)-logit(same) is always about T-1.5, so "previous"
never wins. Variables are placed as identity codes at the end position of depths 10, 11, 12
(the stage the late layers build: the year identity is copied from position 2 by late attention
and combined with the month-order comparison at the end position)."""

def model(x, iv):
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    yr = x[:, 2].astype(int) - 1980
    cuts = {}
    for L in (10, 11, 12):
        cuts[L] = (iv("pm%d" % L, np.eye(144)[m1 * 12 + m2]),
                   iv("py%d" % L, np.eye(240)[m1 * 20 + yr]),
                   iv("qy%d" % L, np.eye(240)[m2 * 20 + yr]))
    # deepest cut whose content differs from what x would give takes precedence
    best = None
    taken = None
    for L in (12, 11, 10):
        a, b, c = cuts[L]
        i1 = a.argmax(1); c1 = i1 // 12; c2 = i1 % 12; cy = c.argmax(1) % 20
        diff = (c1 != m1) | (c2 != m2) | (cy != yr)
        if best is None:
            best = [c1, c2, cy]; taken = diff
        else:
            take = diff & ~taken
            best = [np.where(take, c1, best[0]), np.where(take, c2, best[1]),
                    np.where(take, cy, best[2])]
            taken = taken | diff
    M1, M2, YR = best
    T = np.where(M2 > M1, 1.0, -0.25)
    U = T - 1.5
    return np.stack([np.zeros_like(T), T, U], axis=1)

ALIGN = {}
for _L in (10, 11, 12):
    ALIGN["pm%d" % _L] = ["resid%d@end" % _L]
    ALIGN["py%d" % _L] = ["resid%d@end" % _L]
    ALIGN["qy%d" % _L] = ["resid%d@end" % _L]
