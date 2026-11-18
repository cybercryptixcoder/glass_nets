
import numpy as np

DESCRIPTION = ("The task is the index of the SECOND LARGEST of the 5 inputs. The state is carried forward in "
               "semantic form: the running second-largest value m2 of the values seen so far, the gap m1-m2 to the "
               "running maximum, the gap m2-m3 to the next order statistic, and the sign-split gap x_i-m2 of every "
               "position seen so far (those gaps are what the readout compares). The readout is an asymmetric "
               "weighted distance from m2: out_k = C_k - A_k*relu(x_k-m2) - B_k*relu(m2-x_k), so the position closest "
               "to m2 from below wins. Stage 1 is exact and 2-dimensional: h1 = x1*wx, whose entries have a fixed "
               "sign pattern, so h1 = x1^+*(pos ray) + x1^-*(neg ray).")

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
        s = np.sort(vals, 1)
        m1 = s[:, -1]
        m2 = s[:, -2]
        gap = vals - m2[:, None]
        cols = [m2[:, None], (m1 - m2)[:, None]]
        if t >= 3:
            cols.append((m2 - s[:, -3])[:, None])
        v = iv("v%d" % t, np.concatenate(cols + [np.maximum(gap, 0), np.maximum(-gap, 0)], 1))
        k = 2 + (1 if t >= 3 else 0)
        m2r = v[:, 0]
        g = v[:, k:k + t] - v[:, k + t:k + 2 * t]
        rec = m2r[:, None] + g
        pre = [rec[:, i] for i in range(t)]

    m2 = v[:, 0]
    g = v[:, 3:8] - v[:, 8:13]
    out = (m2[:, None] * 0 + np.array(CC)[None, :] - np.array(AA)[None, :] * np.maximum(g, 0)
           - np.array(BB)[None, :] * np.maximum(-g, 0))
    return out


ALIGN = {"u1": ["h1"], "v2": ["h2"], "v3": ["h3"], "v4": ["h4"], "v5": ["h5"]}
