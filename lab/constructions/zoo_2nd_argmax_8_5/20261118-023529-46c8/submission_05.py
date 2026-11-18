
import numpy as np

DESCRIPTION = ("The network's task is the index of the SECOND LARGEST input, but its running top-2 is seeded by 0 "
               "(h_0 = 0), so it actually takes the second largest of the multiset {0, x_1..x_5}: that is why for "
               "all-negative inputs it answers the maximum (68% of them) and for one positive it answers the largest "
               "negative. Its score for a position is an asymmetric weighted distance from that second-largest value "
               "m2, out_k = c_k - a_k*relu(x_k-m2) - b_k*relu(m2-x_k), so the winning position is essentially the one "
               "closest to m2 (the second largest). The hidden state is carried forward here in the semantic form it "
               "must have: the running second-largest value m2, the gap m1-m2 to the running maximum, and the "
               "sign-split gaps x_i - m2 of every position seen so far; those gaps are exactly what the readout needs. "
               "Stage 1 is exact and 2-dimensional: h1 = x1*wx and wx has a fixed sign pattern, so h1 = x1^+ * (pos "
               "ray) + x1^- * (neg ray).")

A = [3.31, 19.72, 20.50, 17.76, 14.71]
B = [4.13, 8.52, 13.11, 10.64, 10.53]
C = [2.66, 3.12, 4.01, 4.22, 4.64]


def model(x, iv):
    Bn = x.shape[0]
    u1 = iv("u1", np.stack([np.maximum(x[:, 0], 0), np.maximum(-x[:, 0], 0)], 1))
    pre = [u1[:, 0] - u1[:, 1]]

    for t in range(2, 6):
        vals = np.stack(pre, 1)
        Z = np.concatenate([vals, np.zeros((Bn, 1))], 1)
        s = np.sort(Z, 1)
        m1 = s[:, -1]
        m2 = s[:, -2]
        gap = vals - m2[:, None]
        v = iv("v%d" % t, np.concatenate([m2[:, None], (m1 - m2)[:, None],
                                          np.maximum(gap, 0), np.maximum(-gap, 0)], 1))
        m2r = v[:, 0]
        g = v[:, 2:2 + t] - v[:, 2 + t:2 + 2 * t]
        rec = m2r[:, None] + g
        pre = [rec[:, i] for i in range(t)]

    v5 = v
    m2 = v5[:, 0]
    g = v5[:, 2:7] - v5[:, 7:12]
    out = np.array(C)[None, :] - np.array(A)[None, :] * np.maximum(g, 0) - np.array(B)[None, :] * np.maximum(-g, 0)
    return out


ALIGN = {"u1": ["h1"], "v2": ["h2"], "v3": ["h3"], "v4": ["h4"], "v5": ["h5"]}
