import numpy as np
DESCRIPTION = """Step 1 is explained exactly; later stages are not. h1 = relu(x1*W_x) is a 2-dimensional
object: x1 times the positive part of W_x and -x1 times its negative part, so I name those two
numbers at h1. The network's task is 'which position holds the second largest of the ten values'
(it agrees with that rule on 95.3% of inputs = its own accuracy); its residual errors are cases where
the 2nd and 3rd largest are nearly tied (median gap 0.032 on errors vs 0.27 when right) and the
network then keeps the EARLIER position. So the readout is a running top-2 scan that needs a margin
before it displaces a stored value."""

MARGIN = 0.03

def model(x, iv):
    B = x.shape[0]
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    xa = np.abs(x)
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    v1 = np.full(B, -1000000000)
    v2 = np.full(B, -1000000000)
    p1 = np.zeros(B, dtype=int)
    p2 = np.zeros(B, dtype=int)
    t = 0
    while t < 10:
        xt = x[:, t]
        newtop = xt > v1 + MARGIN
        newsec = (xt > v2 + MARGIN) & (v1 >= xt - MARGIN)
        p2 = np.where(newtop, p1, np.where(newsec, t, p2))
        v2 = np.where(newtop, v1, np.where(newsec, xt, v2))
        p1 = np.where(newtop, t, p1)
        v1 = np.where(newtop, xt, v1)
        t += 1
    out = -np.ones((B, 10))
    out[np.arange(B), p2] = 1.0
    return out

ALIGN = {"pos1": ["h1"], "neg1": ["h1"]}
