
import numpy as np

DESCRIPTION = ("The task is the index of the SECOND LARGEST input (relative to a virtual 0, since h_0=0 seeds the "
               "running top-2). The hidden state carries it in semantic form: the running second-largest value m2 of "
               "{0,x_1..x_t}, the gap m1-m2 to the running max, the gap m2-m3 to the next order statistic, and the "
               "sign-split gap x_i-m2 of every position seen so far. The readout is a weighted distance from a target "
               "value T = max(m2, -0.2), out_k = C_k - A_k*relu(x_k-T) - B_k*relu(T-x_k): the position nearest T from "
               "below wins, so it reports the second largest and, when the second largest is the virtual 0 (fewer "
               "than two positive inputs), the value nearest 0. Stage 1 is exact and 2-dimensional: h1 = x1*wx and wx "
               "has a fixed sign pattern, so h1 = x1^+*(pos ray) + x1^-*(neg ray).")

AA = [3.31, 19.72, 20.50, 17.76, 14.71]
BB = [4.13, 8.52, 13.11, 10.64, 10.53]
CC = [2.66, 3.12, 4.01, 4.22, 4.64]
TS = -0.2


def model(x, iv):
    Bn = x.shape[0]
    u1 = iv("u1", np.stack([np.maximum(x[:, 0], 0), np.maximum(-x[:, 0], 0)], 1))
    pre = [u1[:, 0] - u1[:, 1]]

    for t in range(2, 6):
        pre = pre + [x[:, t - 1]]
        vals = np.stack(pre, 1)
        Z = np.concatenate([vals, np.zeros((Bn, 1))], 1)
        s = np.sort(Z, 1)
        m1 = s[:, -1]
        m2 = s[:, -2]
        m3 = s[:, -3]
        gap = vals - m2[:, None]
        v = iv("v%d" % t, np.concatenate([m2[:, None], (m1 - m2)[:, None], (m2 - m3)[:, None],
                                          np.maximum(gap, 0), np.maximum(-gap, 0)], 1))
        m2r = v[:, 0]
        g = v[:, 3:3 + t] - v[:, 3 + t:3 + 2 * t]
        rec = m2r[:, None] + g
        pre = [rec[:, i] for i in range(t)]

    m2v = v[:, 0]
    g = v[:, 3:8] - v[:, 8:13]
    sh = np.maximum(TS - m2v, 0)            # T = m2 + shift
    d = g - sh[:, None]                     # x_k - T
    out = np.array(CC)[None, :] - np.array(AA)[None, :] * np.maximum(d, 0) - np.array(BB)[None, :] * np.maximum(-d, 0)
    return out


ALIGN = {"u1": ["h1"], "v2": ["h2"], "v3": ["h3"], "v4": ["h4"], "v5": ["h5"]}
