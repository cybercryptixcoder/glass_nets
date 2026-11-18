
import numpy as np, math
DESCRIPTION = ("Stage 1: with only ~two large input weights the first hidden state is 2-dimensional: "
               "h1 = relu(x1) * a + relu(-x1) * b, where a,b are the positive and negative parts of the input weight "
               "vector. So the state after step 1 holds just the positive and negative part of x_1. Later stages are "
               "not yet explained (placeholder constant output).")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    x1 = x[:, 0]
    pos = iv("xpos1", np.maximum(x1, 0.0))
    neg = iv("xneg1", np.maximum(-x1, 0.0))
    s = (pos + neg)[:, None]
    return np.zeros((x.shape[0], 10)) + s * 0.0
ALIGN = {"xpos1": ["h1"], "xneg1": ["h1"]}
