
import numpy as np

DESCRIPTION = ("The task is the index of the SECOND LARGEST input. The network keeps a running top-2 in a rectified "
               "difference representation, but the comparisons it makes are inexact: a stored value is compared after "
               "inflation by a factor A (so a NEW value must beat A*old), which makes the tracker prefer the value "
               "already stored and produces the network's characteristic errors in ambiguous cases. The state is "
               "carried here as that tracker's two memories m1 (running max) and m2 (running second max, seeded 0 by "
               "h_0=0) plus the sign-split gap x_i-m2 of every position read so far. The readout is a distance from "
               "the tracker's target T = max(m2, -0.2): out_k = C_k - 30*|x_k - T|, so the position sitting at the "
               "tracked second-largest wins. Stage 1 is exact and 2-dimensional: h1 = x1*wx and wx has a fixed sign "
               "pattern, so h1 = x1^+*(pos ray) + x1^-*(neg ray).")

CC = [2.66, 3.12, 4.01, 4.22, 4.64]
TS = -0.2
ALPHA = 1.10
BETA = 0.95


def track(x, T):
    n = x.shape[0]
    m1 = np.zeros(n)
    m2 = np.zeros(n)
    for t in range(T):
        xt = x[:, t]
        u = xt > ALPHA * m1
        m2 = np.where(u, m1, np.maximum(BETA * m2, xt))
        m1 = np.where(u, xt, ALPHA * m1)
    return m1, m2


def model(x, iv):
    n = x.shape[0]
    u1 = iv("u1", np.stack([np.maximum(x[:, 0], 0), np.maximum(-x[:, 0], 0)], 1))
    pre = [u1[:, 0] - u1[:, 1]]
    v = None
    for t in range(2, 6):
        pre = pre + [x[:, t - 1]]
        vals = np.stack(pre, 1)
        m1, m2 = track(vals, t)
        gap = vals - m2[:, None]
        v = iv("v%d" % t, np.concatenate([m2[:, None], (m1 - m2)[:, None],
                                          np.maximum(gap, 0), np.maximum(-gap, 0)], 1))
        m2r = v[:, 0]
        g = v[:, 2:2 + t] - v[:, 2 + t:2 + 2 * t]
        rec = m2r[:, None] + g
        pre = [rec[:, i] for i in range(t)]

    m2r = v[:, 0]
    g = v[:, 2:7] - v[:, 7:12]
    d = g - np.maximum(TS - m2r, 0)[:, None]
    return np.array(CC)[None, :] - 30.0 * np.abs(d)


ALIGN = {"u1": ["h1"], "v2": ["h2"], "v3": ["h3"], "v4": ["h4"], "v5": ["h5"]}
