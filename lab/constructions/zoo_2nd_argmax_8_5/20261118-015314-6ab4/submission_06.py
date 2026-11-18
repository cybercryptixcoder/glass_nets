import numpy as np

DESCRIPTION = """
Draft 6. THE TASK: the network reports the position of the SECOND LARGEST of the five inputs
(class c = position c+1). Evidence: the decided position holds the 2nd largest value in ~88% of
inputs; on 120 permutations of the well separated multiset {3,1,0,-1,-2} it is right 98.3% of the
time; its errors are near-tie cases (mean top gap 0.27 when it answers with the largest, mean
2nd-3rd gap 0.17 when it answers with the third). Hence class c wins only for x_(c+1) > 0 in the
sign-pattern table.

Step 1 is exact: h1 = relu(x1*W_x) with W_x of fixed sign (units 0,2,4,6,7 positive, units 1,3,5
negative), i.e. h1 = relu(x1)*W_x+ + relu(-x1)*W_x-.

Mechanism sketch under test: the recurrent state carries the running order statistics of the prefix
(m1 = largest, m2 = second largest) in sign-split (positive/negative part) units, plus the
"new maximum" signal relu(x_t - m1_prev) that a max-tracking recurrence needs (the pre-activation of
unit 7 correlates 0.96 with x_t - m1_prev). The readout scores each position as runner-up
(below the max of the others, above the rest). Numbers for stages 2-5 and the readout are placeholders.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def model(x, iv):
    B = x.shape[0]
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]

    m1 = x[:, 0]
    m2 = np.full(B, -np.inf)
    for t in range(1, 5):
        v = x[:, t]
        newm = np.maximum(v - m1, 0.0)
        s = t + 1
        iv("newmax%d" % s, newm)
        hi = m1 + newm
        lo = np.where(v > m1, m1, np.maximum(m2, v))
        m1, m2 = hi, lo
        iv("hi%d" % s, np.maximum(m1, 0.0))
        iv("lo%d" % s, np.maximum(-m1, 0.0))
        iv("hi2_%d" % s, np.maximum(m2, 0.0))
        iv("lo2_%d" % s, np.maximum(-m2, 0.0))

    out = np.zeros((B, 5))
    for c in range(5):
        others = np.delete(x, c, axis=1)
        mx = others.max(1)
        s2 = np.sort(others, 1)[:, -2]
        out[:, c] = (12.0 * np.maximum(x[:, c] - s2, 0.0)
                     - 24.0 * np.maximum(x[:, c] - mx, 0.0)
                     - 6.0 * np.maximum(s2 - x[:, c], 0.0))
    return out + 0.001 * h1.sum(1)[:, None]


ALIGN = {"pos1": ["h1"], "neg1": ["h1"],
         "newmax2": ["h2", "h3", "h4", "h5"],
         "hi2": ["h2", "h3", "h4", "h5"], "lo2": ["h2", "h3", "h4", "h5"],
         "hi2_2": ["h2", "h3", "h4", "h5"], "lo2_2": ["h2", "h3", "h4", "h5"]}
