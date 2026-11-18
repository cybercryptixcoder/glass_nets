import numpy as np

DESCRIPTION = (
    "Two-step ReLU net. Step 1 splits x1 by sign: h1 = [a*relu(x1), b*relu(-x1)]. "
    "Step 2 forms two rectified comparisons of x2 against a multiple of |x1|: "
    "h2_0 = a*relu(x2 - 0.3821*relu(x1) + 2.0852*relu(-x1)), "
    "h2_1 = |b|*relu(-x2 + 1.1415*relu(x1) - 0.7515*relu(-x1)). "
    "Readout is rank-1: class0 wins iff w1 > 0.62*w0."
)

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    p = iv("p", np.maximum(x1, 0.0))
    n = iv("n", np.maximum(-x1, 0.0))
    w0 = iv("w0", np.maximum(x2 - 0.38212356 * p + 2.08516624 * n, 0.0))
    w1 = iv("w1", np.maximum(-x2 + 1.14152943 * p - 0.75152266 * n, 0.0))
    s = iv("s", 0.62 * w0 - w1)
    return np.stack([-s, s], axis=1)

ALIGN = {"p": ["h1"], "n": ["h1"], "w0": ["h2"], "w1": ["h2"]}
