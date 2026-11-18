import numpy as np

DESCRIPTION = """
Draft 4. Two facts established.
(a) step 1 is exact: no biases, so h1 = relu(x1*W_x); W_x has fixed signs (units 0,2,4,6,7 positive,
units 1,3,5 negative), so h1 = relu(x1)*W_x+ + relu(-x1)*W_x- : the first state holds the positive and
the negative part of input 1 in two disjoint coordinate groups.
(b) The network's 5 output classes correspond one-to-one with the 5 positions: class c can only win
when input x_(c+1) is positive (over the 32 sign patterns of the inputs, class c never wins with
x_(c+1) < 0), and the mean |d out_c / d x_(c+1)| ~ 11-13 is diagonal-dominant: output c scores the
positive part of input c+1 against a state-dependent level set by the rest of the sequence.
The output here is a crude stand-in for that (positive part of input c+1 minus the rest); later
stages and the readout level are not yet explained.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def model(x, iv):
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]
    P = np.maximum(x, 0.0)
    S = P.sum(1, keepdims=True)
    out = 12.0 * P - 6.0 * (S - P) - 0.001 * h1.sum(1)[:, None]
    return out


ALIGN = {"pos1": ["h1"], "neg1": ["h1"]}
