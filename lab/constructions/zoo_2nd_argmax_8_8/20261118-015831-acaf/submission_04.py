
import numpy as np

DESCRIPTION = """What the network computes: a linear reservoir with clipping, nothing more. The eight inputs
are independent standard normals. Each step, the new input is injected with a fixed gain into every one of the
eight accumulators and the previous state is mixed by a fixed coupling; the result is clipped at zero (that is
the only nonlinearity, so the network is positively homogeneous: scaling all inputs by c>0 scales the state and
the scores by c and leaves the decision unchanged). There are no comparisons, counts or running maxima anywhere
in the network: every accumulator holds a fixed linear combination of the past inputs, gated by whether its own
pre-activation is positive, and the eight class scores are a fixed linear read-out of the final state.
The accumulators are quoted in a fixed per-unit scale (unit u's state is c_u times the network's hidden unit),
so the gains and couplings below are that account's coefficients, not the raw weight entries.
Stage 1 is the simple case of the same rule: g_1 = relu(x1 * gain) only, since the state starts at zero."""

GAIN = np.array([-0.082919, 0.075402, -0.707443, -1.673094, -0.849465, 0.001046, 3.437996, 0.003974])

M = np.array([
 [0.081661, 2.415879, 1.088362, -36.973387, -2.93757, 0.115479, -4.484341, 1.719164],
 [-0.022110, 0.479736, -0.196991, 0.419167, 0.561988, -0.154780, -0.997470, 0.763648],
 [0.454560, -0.660109, 0.235093, 0.678001, 2.786259, 0.961875, 9.122173, -0.883622],
 [0.044182, 0.439401, -0.030897, -0.671473, -0.031729, 0.028164, 0.083970, 0.010262],
 [-0.070021, -1.188930, -0.145797, 7.814718, 0.272091, -0.007575, 1.641884, -0.225159],
 [-0.265428, -0.177703, -0.579845, 37.811186, -3.739219, 0.785471, 14.920440, 1.902863],
 [0.100789, -0.051591, 0.054295, 0.659586, 0.395475, 0.034142, 0.488446, -0.198628],
 [-0.098470, 1.976962, 0.019676, 1.158094, -0.445536, 0.091067, 10.089078, -0.600398]])

R = np.array([
 [-2.130207, -0.880878, -0.607963, -0.448294, -1.395420, -0.322195, 5.305576, -0.929211],
 [0.271758, 0.045954, -0.020057, -0.089350, -0.502371, -0.086532, 0.145053, 0.050519],
 [-0.272735, 0.121814, -0.231010, 0.262838, 0.907237, 0.284359, 1.120173, -7.564802],
 [0.038156, 0.002079, 0.014082, -0.017941, -0.025892, -0.233389, 0.118289, 0.004032],
 [0.306086, -0.001887, 0.116689, 0.095272, 0.264626, 0.475546, -1.235946, 0.147182],
 [1.655255, 1.035963, 1.724953, 0.852774, -0.344315, 0.757901, -7.687889, 1.066959],
 [-0.077886, -0.017793, -0.080596, 0.018911, 0.127323, 0.019512, 0.004946, -0.017238],
 [-0.003774, 0.011277, 0.067219, -0.658316, 0.372245, 0.029265, -0.364052, 0.012961]])

def model(x, iv):
    s = np.zeros((x.shape[0], 8))
    for t in range(8):
        s = iv("g%d" % (t + 1), np.maximum(x[:, t:t+1] * GAIN[None, :] + s @ M, 0.0))
    return s @ R

ALIGN = {"g%d" % t: ["h%d" % t] for t in range(1, 9)}
