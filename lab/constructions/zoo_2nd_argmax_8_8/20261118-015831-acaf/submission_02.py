
import numpy as np

DESCRIPTION = """Partial construction. Stage 1 (h1) is exact: with no biases the network is
positively homogeneous, and the first hidden state is h1 = relu(x1 * W_x), i.e. it carries only the sign and
magnitude of the first input. Two variables (positive and negative part of x1) reproduce h1 exactly.
For the remaining stages the file only carries a rough first-order (linear) read-out of the inputs, used as a
placeholder so that the output is not constant; the real mechanism is a relu recursion whose piecewise-linear
structure is still being identified."""

WX = np.array([-0.207, 0.03, -1.203, -0.151, -0.467, 0.003, 0.619, 0.004])
WX_POS = np.maximum(WX, 0.0)
WX_NEG = np.minimum(WX, 0.0)

GLIN = np.array([
 [ 4.70,  0.19, -0.09, -0.34, -0.18,  0.17,  0.15,  0.15],
 [ 0.83,  7.19, -0.21, -0.15, -0.15,  0.18,  0.25,  0.30],
 [-0.19, -0.33,  6.53, -1.24, -1.08, -0.92, -0.83, -0.75],
 [ 0.18,  0.26, -0.09,  6.73, -0.64, -0.36, -0.53, -0.40],
 [-0.08,  0.08, -0.09, -0.30,  5.86, -0.53, -0.59, -0.69],
 [-0.15, -0.01, -0.10, -0.08, -0.27,  4.62, -0.70, -0.62],
 [-0.56, -0.50, -0.54, -0.40, -0.55, -0.39,  3.84, -1.23],
 [-0.24, -0.03, -0.25, -0.13, -0.12, -0.07, -0.22,  4.00]])

def model(x, iv):
    pos = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    neg = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * WX_POS[None, :] + neg[:, None] * WX_NEG[None, :]
    out = x @ GLIN
    return out

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
