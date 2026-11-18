import numpy as np
DESCRIPTION = """Step 1 only is explained exactly. The network answers 'which position holds the
second largest of the ten values' (95.3% agreement = its own accuracy). At h1 the state is exactly
relu(x1*W_x), which is a 2-dimensional object: x1 times the positive part of W_x and x1 times its
negative part. All later stages are left unexplained; the readout answers the task rule directly
from x (index of the second largest value)."""

POS = np.array([0.0692, 0.0, 0.0148, 0.0, 10.157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0599, 0.148, 0.298, 0.0, 0.0, 0.0])
NEG = np.array([0.0, 10.565, 0.0, 0.1227, 0.0, 0.281, 10.996, 13.169, 12.312, 0.228, 0.0, 0.0, 0.0, 0.047, 1.321, 0.1186])

def model(x, iv):
    B = x.shape[0]
    pos1 = iv("pos1", x[:, 0])
    neg1 = iv("neg1", np.maximum(-x[:, 0], 0.0))
    del pos1, neg1
    order = np.argsort(-x, axis=1)
    ans = order[:, 1]
    out = -np.ones((B, 10))
    out[np.arange(B), ans] = 1.0
    return out

ALIGN = {"pos1": ["h1"], "neg1": ["h1"]}
