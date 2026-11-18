import numpy as np

DESCRIPTION = """The first hidden state is exactly relu(x1*wx); since wx has mixed signs it equals
p*wx_pos + n*wx_neg with p = relu(x1), n = relu(-x1) -- i.e. two numbers (the positive and the
negative part of the first input) times two fixed 8-dim patterns. From there the state is
amplified by the recurrence (spectral radius ~3.5) and the readout difference between classes
6 and 7 is nearly orthogonal to the amplified direction, so the decision is the sign of a
delicately balanced evidence score. To first order that score is a weighted sum of the inputs
whose weights fall off geometrically with the step (each later input is amplified by fewer
recurrence steps): about (-50,-20,-7,-3,+0.5,-0.9,+0.6,+0.5) x_t. Only classes 6 and 7 are
ever competitive (their difference is the score; class 7 wins when the score is positive)."""

WX = np.array([-1.2027, -0.4672, -0.1506, 0.6188, 0.0032, -0.2073, 0.0302, 0.0040])
WXP = np.maximum(WX, 0.0)
# evidence direction in input space: class 7 wins when sum_t ev[t]*x_t > 0
EV = np.array([1.3614, 0.6840, 0.3391, 0.1394, 0.0069, 0.0103, 0.0207, -0.0009])


def model(x, iv):
    B = x.shape[0]
    p1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    score = iv("score0", np.zeros(B))
    # the first input's contribution, read from the state variables so that an
    # intervention on the step-1 state moves the score the way the network does
    x1_eff = p1 - n1
    s = iv("score1", score + EV[0] * x1_eff)
    for t in range(1, 8):
        s = iv("score%d" % (t + 1), s + EV[t] * x[:, t])
    out = np.zeros((B, 8))
    out[:, 6] = -np.abs(s) * 0.0 - s
    out[:, 7] = s
    return out


ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
