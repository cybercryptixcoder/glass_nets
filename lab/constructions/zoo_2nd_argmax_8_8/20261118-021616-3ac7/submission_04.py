import numpy as np

DESCRIPTION = """Step 1 is exact: h1 = relu(x1*wx) = p*wx_pos + n*wx_neg with p = relu(x1), n = relu(-x1),
so the step-1 state holds exactly two numbers: the positive and the negative part of the first
input, each scaled by a fixed 8-dim pattern. From there the recurrence amplifies the state by a
factor of ~3.5 per step along one dominant direction, while the readout difference between the
only two competitive classes (6 and 7) is nearly orthogonal to that dominant direction. The
decision is therefore the sign of a finely balanced evidence score: a mostly-linear weighted sum
of the inputs whose weights decay with the step (an earlier input is amplified by more recurrence
steps), corrected by the rectification (only the positive/negative part of each input is passed
on, which is why each input enters once linearly and once through its two half-wave rectified
parts). Inputs at later steps barely matter. Class 7 wins when the score is positive."""

WX = np.array([-1.2027, -0.4672, -0.1506, 0.6188, 0.0032, -0.2073, 0.0302, 0.0040])
# evidence weights: direct part, positive-part correction, negative-part correction
WL = np.array([-7.58, -3.80, -1.89, -0.76, -0.27, -0.17, 0.02, 0.00])
WP = np.array([-3.72, 0.35, -1.35, -0.48, -0.23, -0.12, -0.02, -0.03])
WN = np.array([3.85, 4.15, 0.54, 0.27, 0.04, 0.05, -0.04, -0.03])


def model(x, iv):
    B = x.shape[0]
    p1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    x1 = iv("x1", p1 - n1)
    u = np.stack([x1] + [x[:, t] for t in range(1, 8)], axis=1)
    pos = iv("pos_parts", np.maximum(u, 0.0))
    neg = iv("neg_parts", np.maximum(-u, 0.0))
    level = iv("level", np.zeros(B))
    s = iv("lin", level + u @ WL)
    s = iv("score", s + pos @ WP + neg @ WN)
    out = np.zeros((B, 8))
    out[:, 6] = -s
    out[:, 7] = s
    return out


ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
