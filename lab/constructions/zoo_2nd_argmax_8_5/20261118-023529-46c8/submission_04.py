
import numpy as np

DESCRIPTION = ("The network's task is the index of the SECOND LARGEST of the five inputs; it gets 88.9%, confusing it "
               "with the maximum (5%) or the third value (6%) only when the gaps are small. Its output scores a "
               "position by a weighted asymmetric distance from the running second-largest value m2: "
               "out_k = c_k - a_k*relu(x_k-m2) - b_k*relu(m2-x_k) (per-class slopes measured by regression). "
               "Hypothesis under test for the hidden state: it stores the sign-split gaps of every prefix value to "
               "the running second-largest value, together with the sign-split gap between m1 and m2; stage 1 is "
               "exact (h1 = x1*wx, whose weights split by sign, so h1 lies in the 2-D cone of x1^+ and x1^-).")

A = [3.31, 19.72, 20.50, 17.76, 14.71]
B = [4.13, 8.52, 13.11, 10.64, 10.53]
C = [2.66, 3.12, 4.01, 4.22, 4.64]


def model(x, iv):
    u1 = iv("u1", np.stack([np.maximum(x[:, 0], 0), np.maximum(-x[:, 0], 0)], 1))
    x1 = u1[:, 0] - u1[:, 1]
    xx = np.concatenate([x1[:, None], x[:, 1:]], 1)

    m2 = np.zeros(len(xx))
    for t in range(1, 6):
        pre = xx[:, :t]
        s = np.sort(pre, 1)
        m2 = s[:, -2] if t >= 2 else np.zeros(len(xx))
        m1 = s[:, -1]
        feats = []
        if t == 1:
            feats = [np.maximum(xx[:, 0], 0), np.maximum(-xx[:, 0], 0)]
        else:
            for i in range(t):
                g = xx[:, i] - m2
                feats.append(np.maximum(g, 0))
                feats.append(np.maximum(-g, 0))
            d = m1 - m2
            feats.append(np.maximum(d, 0))
            feats.append(np.maximum(-d, 0))
        v = iv("s%d" % t, np.stack(feats, 1))
        if t == 5:
            gaps = v

    g = xx - m2[:, None]
    out = np.array(C)[None, :] - np.array(A)[None, :] * np.maximum(g, 0) - np.array(B)[None, :] * np.maximum(-g, 0)
    return out


ALIGN = {"u1": ["h1"], "s1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
