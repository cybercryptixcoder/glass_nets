import numpy as np

DESCRIPTION = """
Stage 'step 1' (h1). The input weight row has a fixed sign pattern (units 1,3,5 negative, units
0,2,4,6,7 positive) and there are no biases, so h1 = relu(x1)*W_x+ + relu(-x1)*W_x- exactly: the
first state is a sign-magnitude code of input 1, the positive part in the positive-weight units and
the negative part in the negative-weight units. Two numbers, pos1 and neg1, are the whole content
of h1 (x1 = pos1 - neg1).
Provisional account of the rest: the state carries the running order statistics of the inputs read
so far (largest m and second largest s of the prefix); the readout scores each position c by how
far x_c rises above the second largest of the four other inputs, minus a steeper penalty for rising
above their largest, minus a penalty for each unit it lies below the second largest - i.e. a
position wins when it is above everything except one other input (it holds the second largest).
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def order_out(x):
    out = np.zeros((x.shape[0], 5))
    for c in range(5):
        others = np.delete(x, c, axis=1)
        mo = np.max(others, axis=1)
        so = np.sort(others, axis=1)[:, -2]
        xc = x[:, c]
        out[:, c] = (10.0 * np.maximum(xc - so, 0.0)
                     - 22.0 * np.maximum(xc - mo, 0.0)
                     - 9.0 * np.maximum(so - xc, 0.0))
    return out


def model(x, iv):
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]
    x1 = pos - neg
    seq = np.concatenate([x1[:, None], x[:, 1:]], axis=1)
    return order_out(seq)


ALIGN = {"pos1": ["h1"], "neg1": ["h1"]}
