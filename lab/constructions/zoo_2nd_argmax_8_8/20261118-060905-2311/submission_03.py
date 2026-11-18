
import numpy as np

DESCRIPTION = ("Faithful recurrence h_t = relu(x_t * W_x + h_{t-1} @ W_h) with the measured weights. "
               "Structurally: the input weights are nonzero only on units {0,1,2,3,4,6} (units 5 and 7 "
               "receive essentially no input, so their 0.003/0.004 weights are dropped). Units {1,6} are "
               "driven by x_t^+, units {0,2,3,4} by x_t^-, so the state splits positive and negative input "
               "parts. From step ~4 on almost every unit is active, so the late recurrence is a linear "
               "rotation-like map (W_h has complex eigenvalue pairs at angles ~39, 85, 133 deg) and the "
               "readout is a linear comparison of the 8 class scores; class k is driven mainly by x_{k+1}. "
               "Still a re-description: the recurrent and readout matrices are copied.")

WX = [-0.207, 0.030, -1.203, -0.151, -0.467, 0.0, 0.619, 0.0]
WH = [[0.082, 0.387, 0.740, -1.331, -0.646, 0.143, -0.323, 0.688],
      [-0.138, 0.480, -0.837, 0.094, 0.773, -1.200, -0.449, 1.909],
      [0.668, -0.155, 0.235, 0.036, 0.901, 1.754, 0.966, -0.520],
      [1.227, 1.953, -0.584, -0.671, -0.194, 0.970, 0.168, 0.114],
      [-0.318, -0.865, -0.451, 1.279, 0.272, -0.043, 0.537, -0.409],
      [-0.214, -0.023, -0.318, 1.098, -0.663, 0.785, 0.866, 0.614],
      [1.400, -0.115, 0.513, 0.330, 1.208, 0.588, 0.488, -1.103],
      [-0.246, 0.791, 0.033, 0.104, -0.245, 0.282, 1.816, -0.600]]
WO = [[-0.852, -0.352, -0.243, -0.179, -0.558, -0.129, 2.122, -0.372],
      [0.679, 0.115, -0.050, -0.223, -1.256, -0.216, 0.363, 0.126],
      [-0.160, 0.072, -0.136, 0.155, 0.534, 0.167, 0.659, -4.450],
      [0.424, 0.023, 0.156, -0.199, -0.288, -2.593, 1.314, 0.045],
      [0.557, -0.003, 0.212, 0.173, 0.481, 0.865, -2.247, 0.268],
      [0.534, 0.334, 0.556, 0.275, -0.111, 0.244, -2.480, 0.344],
      [-0.433, -0.099, -0.448, 0.105, 0.707, 0.108, 0.027, -0.096],
      [-0.004, 0.011, 0.067, -0.658, 0.372, 0.029, -0.364, 0.013]]

wx = np.array(WX); wh = np.array(WH); wo = np.array(WO)

def model(x, iv):
    h = np.zeros((x.shape[0], 8))
    for t in range(8):
        h = iv("h%d" % (t + 1), np.maximum(x[:, t:t + 1] * wx[None, :] + h @ wh, 0.0))
    return h @ wo

ALIGN = {"h%d" % t: ["h%d" % t] for t in range(1, 9)}
