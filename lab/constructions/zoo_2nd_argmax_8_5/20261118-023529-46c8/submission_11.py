
import numpy as np

DESCRIPTION = ("The task is the index of the SECOND LARGEST of the five inputs. The helper state is carried in "
               "semantic form: the running second-largest value m2 of the values read so far, the gap m1-m2 to the "
               "running maximum, the gap m2-m3 to the next order statistic, and the sign-split gap x_i-m2 of every "
               "position read so far. The readout is a distance from a target value T = max(m2, -0.2): "
               "out_k = C_k - 30*|x_k - T|, so the position whose value sits at the second-largest wins; when the "
               "second largest is far below zero (fewer than two positive inputs) the target is pinned near 0 and the "
               "value nearest zero wins. Stage 1 is exact and 2-dimensional: h1 = x1*wx and wx has a fixed sign "
               "pattern, so h1 = x1^+*(pos ray) + x1^-*(neg ray).")

CC = [2.66, 3.12, 4.01, 4.22, 4.64]
TS = -0.2


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

    m2r = v[:, 0]
    g = v[:, 3:8] - v[:, 8:13]
    shift = np.maximum(TS - m2r, 0)
    d = g - shift[:, None]
    out = np.array(CC)[None, :] - 30.0 * np.abs(d)
    return out


ALIGN = {"u1": ["h1"], "v2": ["h2"], "v3": ["h3"], "v4": ["h4"], "v5": ["h5"]}
