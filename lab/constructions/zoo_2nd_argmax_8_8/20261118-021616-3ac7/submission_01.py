import numpy as np

DESCRIPTION = """A one-layer ReLU RNN over 8 steps, no biases, homogeneous of degree 1.
h_t = relu(x_t * wx + h_{t-1} @ wh). The input scalar at step t only enters through
the fixed 8-vector wx, so each step adds the same 8-dim direction scaled by x_t, then
mixes the previous state through wh and rectifies. The state's growth is dominated by a
single direction; after several steps the units co-vary almost perfectly (rank-1 growth),
so the decision is essentially the sign of an accumulated, thresholded weighted sum of
recent inputs, with early inputs weighted much more heavily because they were amplified
by the recurrence for more steps. The readout projects the final state onto 8 class
directions; only classes 6 and 7 ever win (they are nearly opposite directions)."""

WX = np.array([-1.2027, -0.4672, -0.1506, 0.6188, 0.0032, -0.2073, 0.0302, 0.0040])
WH = np.array([
 [-0.6715, -0.6004, 0.2721, -1.1035, 1.0977, -0.6463, -0.1382, -0.4506],
 [-0.5198, 0.6138, 0.9701, 0.0335, 0.9014, -0.6634, 0.0359, 0.6685],
 [-0.8372, -0.0229, 0.7401, -0.1939, 0.1679, 1.2084, 0.0943, -0.1553],
 [0.5128, 0.8663, 0.2351, 1.2788, -1.3310, 0.3865, 1.8160, -0.8647],
 [1.2273, 1.7540, 0.0817, -0.3229, 0.9659, -0.5836, -0.0427, -0.3180],
 [0.5880, 1.9091, -0.3183, 1.9529, 0.2823, 0.4797, 0.5373, 0.4884],
 [-0.2450, 0.3298, -0.2141, 1.3999, -1.1995, 0.7727, 0.6877, 0.7855],
 [-0.4094, 0.1140, 0.1432, -0.1146, 0.7908, -0.4489, -0.2462, 0.1042]])
WO = np.array([
 [-0.6583, 0.7073, 0.1051, -0.1604, 0.1084, -0.5582, 0.5564, 0.3342],
 [-0.4327, 0.2676, -0.0988, -0.1289, 0.0130, -0.0034, 0.3722, 0.1673],
 [0.8646, -2.4800, -0.3641, 1.3143, 0.0448, -0.8521, -0.2163, 0.3626],
 [-0.1359, 0.0717, -0.2877, 0.0672, -0.4478, -2.5932, 0.1732, 0.0275],
 [-0.3524, -0.1793, -0.2432, -1.2559, -0.2234, 0.1263, 0.4240, -0.3717],
 [-4.4499, -0.0958, 0.2445, -2.2472, 0.4811, 0.5337, 0.0113, 0.2122],
 [-0.0501, -0.1993, 0.5565, 0.0231, 0.2751, 0.1565, 0.6794, 0.5340],
 [-0.1111, 0.1149, 0.3442, 0.0293, 0.6589, 0.1546, -0.0038, 2.1222]])


def model(x, iv):
    h = np.zeros((x.shape[0], 8))
    for t in range(8):
        pre = x[:, t:t+1] * WX[None, :] + h @ WH
        h = iv("h%d" % (t + 1), np.maximum(pre, 0.0))
    out = h @ WO
    return out


ALIGN = {"h%d" % t: ["h%d" % t] for t in range(1, 9)}
