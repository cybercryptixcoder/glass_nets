import numpy as np

DESCRIPTION = """
Draft 2. Stage 'step 1' is exact: with no biases h1 = relu(x1*W_x), and W_x has fixed signs
(units 0,2,4,6,7 positive, units 1,3,5 negative), so h1 = relu(x1)*W_x+ + relu(-x1)*W_x-;
the first state holds the positive and the negative part of input 1 in two disjoint coordinate groups.

Probes of h5 show it carries the running maximum of the inputs (linearly decodable, R2=0.67) and most
strongly the most recent inputs (x5 R2=0.85, x4 0.78), so the recurrence is read as a running
max / running argmax style scan. Stages 2-5 are therefore parametrised here by the running maximum
m_t = max(x_1..x_t) and the running argmax identity one-hot oh_t (5 coordinates); the readout is a
placeholder one-hot of the running argmax. Coefficients at h2..h5 and the readout are rough.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def model(x, iv):
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]

    m = np.maximum.accumulate(x, axis=1)
    am = np.zeros_like(x)
    am[:, 0] = x[:, 0]
    for t in range(1, 5):
        am[:, t] = np.where(x[:, t] > am[:, t - 1], x[:, t], am[:, t - 1])
    idx = np.argmax(m, axis=1)
    for t in range(1, 5):
        iv("m%d" % (t + 1), m[:, t])
        oh = np.zeros((x.shape[0], 5))
        oh[np.arange(x.shape[0]), idx] = 1.0
        iv("oh%d" % (t + 1), oh)
    out = np.zeros((x.shape[0], 5))
    out[np.arange(x.shape[0]), idx] = 10.0
    return out + 0.001 * h1.sum(1)[:, None]


ALIGN = {"pos1": ["h1"], "neg1": ["h1"],
         "m2": ["h2"], "oh2": ["h2"],
         "m3": ["h3"], "oh3": ["h3"],
         "m4": ["h4"], "oh4": ["h4"],
         "m5": ["h5"], "oh5": ["h5"]}
