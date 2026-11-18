
import numpy as np

DESCRIPTION = ("Task: the index of the SECOND LARGEST input, with the running top-2 seeded by 0 (h_0=0), so it is "
               "the second largest of {0,x_1..x_5}; the score for position k is an asymmetric weighted distance from "
               "that second-largest value m2. The state is carried in semantic form: running m2, the gap m1-m2 to the "
               "running max, the gap m2-m3 to the next-lower order statistic, and the sign-split gaps x_i-m2 of each "
               "position so far. Stage 1 is exact: h1 = x1*wx = x1^+*(pos ray) + x1^-*(neg ray).")

AA = [3.31, 19.72, 20.50, 17.76, 14.71]
BB = [4.13, 8.52, 13.11, 10.64, 10.53]
CC = [2.66, 3.12, 4.01, 4.22, 4.64]


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

    g = v[:, 3:8] - v[:, 8:13]
    out = np.array(CC)[None, :] - np.array(AA)[None, :] * np.maximum(g, 0) - np.array(BB)[None, :] * np.maximum(-g, 0)
    return out


ALIGN = {"u1": ["h1"], "v2": ["h2"], "v3": ["h3"], "v4": ["h4"], "v5": ["h5"]}
