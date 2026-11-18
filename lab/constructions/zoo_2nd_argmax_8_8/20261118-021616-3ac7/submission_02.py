import numpy as np

DESCRIPTION = """Stage 1 is exact: the first hidden state is relu(x1*wx) where wx has mixed signs,
so h1 = p*wx_pos + n*wx_neg with p = relu(x1), n = relu(-x1) -- the state at step 1 is the
magnitude |x1| split into two fixed 8-dim sign patterns. Later steps add the same 8-dim input
direction scaled by x_t (again split into positive and negative parts) and mix the previous
state through the recurrence, so the state keeps growing roughly in proportion to an
accumulated, sign-split weighted sum of the inputs, with the earliest inputs weighted far
more heavily (they are amplified by more recurrence steps). The readout makes only classes
6 and 7 competitive; their difference is one linear functional of the final state, so the
decision is the sign of an accumulated evidence score."""

WX = np.array([-1.2027, -0.4672, -0.1506, 0.6188, 0.0032, -0.2073, 0.0302, 0.0040])
WXP = np.maximum(WX, 0.0)
WXN = np.minimum(WX, 0.0)
# direction in input space along which the accumulated evidence grows (class 7 vs class 6)
D = np.array([1.3614, 0.6840, 0.3391, 0.1394, 0.0069, 0.0103, 0.0207, -0.0009])


def model(x, iv):
    B = x.shape[0]
    p1 = iv("h1_pos", np.maximum(x[:, 0], 0.0))
    n1 = iv("h1_neg", np.maximum(-x[:, 0], 0.0))
    s = np.zeros(B)
    for t in range(1, 8):
        s = iv("s%d" % (t + 1), s + x[:, t] * D[t])
    score = p1 * D[0] + n1 * (-D[0]) * 0.0
    score = iv("score", score + s)
    out = np.zeros((B, 8))
    out[:, 6] = -score
    out[:, 7] = score
    return out


ALIGN = {"h1_pos": ["h1"], "h1_neg": ["h1"]}
