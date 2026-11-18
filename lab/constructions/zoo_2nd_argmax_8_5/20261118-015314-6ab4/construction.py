import numpy as np

DESCRIPTION = """
What the network does: it reads 5 independent numbers and scores the five positions; the winning
position is (88.9% of the time) the position of the SECOND LARGEST input. Its errors are the cases
where it answers the largest or the third largest value instead, i.e. the top of the order
statistics is only approximately resolved.

How it is computed, stage by stage.
Step 1 (h1, explained exactly). The input row W_x has a fixed sign pattern: units 0,2,4,6,7 have
W_x > 0 and units 1,3,5 have W_x < 0, and there are no biases, so
     h1 = relu(x1)*W_x+ + relu(-x1)*W_x-
i.e. the first state is a sign-magnitude code of input 1: its positive part is written, with the
per-unit gains |W_x|, into the positive-weight units and its negative part into the negative-weight
units. pos1 = relu(x1) and neg1 = relu(-x1) are the whole content of h1 (x1 = pos1 - neg1).

Later steps (provisional). The recurrence makes each large-|W_x| unit a threshold comparator against
a threshold carried by the state: unit 7 fires on relu(x_t - (running largest of the prefix)) and
units 1,3,5 fire on relu((running largest) - x_t) at three different gains, so the state keeps the
running order statistics of the prefix - the largest m and the second largest s of the numbers read
so far - in sign-magnitude form; the small-|W_x| units 0,2,4,6 carry the memory term.

Readout. Each position c is scored by how far its value rises above the second largest of the other
four inputs, with a steeper penalty for rising above their largest and a penalty for lying below
their second largest, so that the position holding the second largest value wins.
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
