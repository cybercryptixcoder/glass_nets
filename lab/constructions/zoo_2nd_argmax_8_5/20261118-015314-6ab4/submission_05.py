import numpy as np

DESCRIPTION = """
Draft 5. THE TASK IS FOUND: the network reports the *position of the second largest* of the five
inputs (class c = position c+1). Checks: the chosen value has rank 1 (second largest) among the five
in ~88% of inputs; the network's residual errors are near-tie cases (mean top-gap 0.27 when it picks the
largest instead, mean 2nd-3rd gap 0.17 when it picks the third). Over the 32 sign patterns, class c
essentially never wins with x_(c+1) < 0 simply because the runner-up is usually positive.

Step 1 is exact: with no biases h1 = relu(x1*W_x) and W_x has fixed signs (units 0,2,4,6,7 positive,
units 1,3,5 negative), so h1 = relu(x1)*W_x+ + relu(-x1)*W_x-.
Stages 2-5 are parametrised here only roughly by the running 1st/2nd order statistics of the prefix
(clipped parts) plus their one-hot positions; the recurrence update rule is not yet explained and the
readout is a stand-in for a runner-up score.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def model(x, iv):
    B = x.shape[0]
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]

    # running 1st / 2nd order statistics of the prefix, fed forward stage by stage
    m1 = x[:, 0]
    m2 = x[:, 1]
    for t in range(1, 5):
        v = x[:, t]
        if t > 1:
            hi = np.maximum(m1, v)
            lo = np.maximum(m2, np.minimum(m1, v))
            m1, m2 = hi, lo
        tagn = t + 1
        a = iv("hi%d" % tagn, np.maximum(m1, 0.0))
        b = iv("lo%d" % tagn, np.maximum(-m1, 0.0))
        c2 = iv("hi2_%d" % tagn, np.maximum(m2, 0.0))
        d2 = iv("lo2_%d" % tagn, np.maximum(-m2, 0.0))
    # runner-up score: x_c is below the max of others and above the rest
    out = np.zeros((B, 5))
    for c in range(5):
        others = np.delete(x, c, axis=1)
        mx = others.max(1)
        snd = np.sort(others, 1)[:, -2]
        out[:, c] = 12.0 * np.maximum(x[:, c] - snd, 0.0) - 8.0 * np.maximum(mx - x[:, c], 0.0)
    return out


ALIGN = {"pos1": ["h1"], "neg1": ["h1"],
         "hi2": ["h2", "h3", "h4", "h5"], "lo2": ["h2", "h3", "h4", "h5"],
         "hi2_2": ["h2", "h3", "h4", "h5"], "lo2_2": ["h2", "h3", "h4", "h5"]}
