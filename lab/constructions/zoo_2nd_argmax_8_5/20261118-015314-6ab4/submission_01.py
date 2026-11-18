import numpy as np

DESCRIPTION = """
Stage 'step 1' only (first draft). The input weight row W_x is split by sign: units 0,2,4,6,7
have W_x>0 and so are driven only by positive x1; units 1,3,5 have W_x<0 and so are driven only
by negative x1. Because there are no biases, h1 = relu(x1*W_x) is exactly
    h1 = relu(x1)*W_x+ + relu(-x1)*W_x-
with W_x+ = max(W_x,0), W_x- = max(-W_x,0): i.e. the first hidden state carries the positive part
and the negative part of the first input, in two disjoint coordinate groups (sign-magnitude code).
Later stages (h2..h5 and the readout) are not explained yet; the output here is a placeholder.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def model(x, iv):
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]
    # placeholder readout (stages h2..h5 not explained yet)
    s = h1.sum(1)
    return np.repeat(s[:, None], 5, axis=1)


ALIGN = {"pos1": ["h1"], "neg1": ["h1"]}
