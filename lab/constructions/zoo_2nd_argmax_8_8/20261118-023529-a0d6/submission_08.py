
import numpy as np

DESCRIPTION = (
 "Mechanism: a bank of 8 leaky integrators with a rectifying clip, reset by one scalar per step. "
 "Step t: each unit j receives the scalar x_t through a fixed input gain Wx[j] plus the mixed previous state "
 "(h_{t-1} @ Wh)[j], and keeps it only if the total is positive (clip at zero), so the state is a "
 "sign-gated running sum of the whole input stream. W_h amplifies the state about 1.5x per step, so a late "
 "input contributes far more than an early one; the clip makes each unit switch on/off with the sign pattern of "
 "the prefix, which is why the same input coordinate can help or hurt a class depending on the rest of the prefix. "
 "The output compares 8 fixed directions of the final state: class c is scored by projecting h_8 on W_o[:,c], "
 "and the decision is the largest score. Nothing else (no bias, no per-step parameters) enters: the same Wx, Wh, Wo "
 "are reused every step, so the network is a single clipped linear system driven 8 times."
)

Wx = np.array([-0.2073, 0.0302, -1.2027, -0.1506, -0.4672, 0.0032, 0.6188, 0.0040])

Wh = np.array([
 [ 0.0817,  0.3865,  0.7401, -1.3310, -0.6463,  0.1432, -0.3229,  0.6877],
 [-0.1382,  0.4797, -0.8372,  0.0943,  0.7727, -1.1995, -0.4489,  1.9091],
 [ 0.6685, -0.1553,  0.2351,  0.0359,  0.9014,  1.7540,  0.9659, -0.5198],
 [ 1.2273,  1.9529, -0.5836, -0.6715, -0.1939,  0.9701,  0.1679,  0.1140],
 [-0.3183, -0.8647, -0.4506,  1.2788,  0.2721, -0.0427,  0.5373, -0.4094],
 [-0.2141, -0.0229, -0.3180,  1.0977, -0.6634,  0.7855,  0.8663,  0.6138],
 [ 1.3999, -0.1146,  0.5128,  0.3298,  1.2084,  0.5880,  0.4884, -1.1035],
 [-0.2462,  0.7908,  0.0335,  0.1042, -0.2450,  0.2823,  1.8160, -0.6004]])

# readout written as differences from class 0: argmax is unchanged, out_0 = 0
Wd = np.array([
 [-0.8521,  0.6794, -0.1604,  0.4240,  0.5565,  0.5340, -0.4327, -0.0038],
 [-0.3524,  0.1149,  0.0717,  0.0231, -0.0034,  0.3342, -0.0988,  0.0113],
 [-0.2432, -0.0501, -0.1359,  0.1565,  0.2122,  0.5564, -0.4478,  0.0672],
 [-0.1793, -0.2234,  0.1546, -0.1993,  0.1732,  0.2751,  0.1051, -0.6583],
 [-0.5582, -1.2559,  0.5337, -0.2877,  0.4811, -0.1111,  0.7073,  0.3722],
 [-0.1289, -0.2163,  0.1673, -2.5932,  0.8646,  0.2445,  0.1084,  0.0293],
 [ 2.1222,  0.3626,  0.6589,  1.3143, -2.2472, -2.4800,  0.0275, -0.3641],
 [-0.3717,  0.1263, -4.4499,  0.0448,  0.2676,  0.3442, -0.0958,  0.0130]])

def model(x, iv):
    h = np.zeros((x.shape[0], 8))
    for t in range(8):
        h = iv("h%d" % (t + 1), np.maximum(h @ Wh + x[:, t:t + 1] * Wx, 0.0))
    s = h @ Wd                   # s[:,c] = out_c - out_0
    return s - s[:, :1]          # same ordering as the network's 8 scores

ALIGN = {"h%d" % (t + 1): ["h%d" % (t + 1)] for t in range(8)}
