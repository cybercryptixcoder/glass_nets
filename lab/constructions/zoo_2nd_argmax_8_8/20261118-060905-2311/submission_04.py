
import numpy as np

DESCRIPTION = ("Faithful recurrence h_t = relu(x_t*W_x + h_{t-1}@W_h), with two exact invariances used to "
               "write fewer numbers: (i) the input weights are nonzero only on units {0,1,2,3,4,6}; "
               "(ii) the class decision is unchanged when the same amount is added to all class scores, "
               "so each row of the readout may be shifted by the amount in class 0 - that column is then "
               "identically zero and is not written, leaving a 7-column readout. Units {1,6} are driven by "
               "x^+, units {0,2,3,4} by x^-, so the state splits positive and negative input parts; from "
               "step 4 on nearly all units are active, so the tail is a linear rotation-like map whose "
               "readout is a set of 8 linear class scores (class k mostly reads x_{k+1}).")

WX = [-0.207, 0.030, -1.203, -0.151, -0.467, 0.0, 0.619, 0.0]
WH = [[0.082, 0.387, 0.740, -1.331, -0.646, 0.143, -0.323, 0.688],
      [-0.138, 0.480, -0.837, 0.094, 0.773, -1.200, -0.449, 1.909],
      [0.668, -0.155, 0.235, 0.036, 0.901, 1.754, 0.966, -0.520],
      [1.227, 1.953, -0.584, -0.671, -0.194, 0.970, 0.168, 0.114],
      [-0.318, -0.865, -0.451, 1.279, 0.272, -0.043, 0.537, -0.409],
      [-0.214, -0.023, -0.318, 1.098, -0.663, 0.785, 0.866, 0.614],
      [1.400, -0.115, 0.513, 0.330, 1.208, 0.588, 0.488, -1.103],
      [-0.246, 0.791, 0.033, 0.104, -0.245, 0.282, 1.816, -0.600]]
# readout differences: row_i - row_0, classes 1..7 (class 0 is identically zero)
WOD = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.467, 0.193, -0.044, -0.698, -0.087, -1.759, 0.498],
       [0.424, 0.107, 0.334, 1.092, 0.296, -1.463, -4.078],
       [0.375, 0.399, -0.020, 0.270, -2.464, -0.808, 0.417],
       [0.349, 0.455, 0.352, 1.039, 0.994, -4.369, 0.640],
       [0.686, 0.799, 0.454, 0.447, 0.373, -4.602, 0.716],
       [0.253, -0.205, 0.284, 1.265, 0.237, -2.095, 0.276],
       [0.363, 0.310, -0.479, 0.930, 0.158, -2.486, 0.385]]

wx = np.array(WX)
wh = np.array(WH)
wo = np.zeros((8, 8))
wo[:, 1:] = np.array(WOD)

def model(x, iv):
    h = np.zeros((x.shape[0], 8))
    for t in range(8):
        h = iv("h%d" % (t + 1), np.maximum(x[:, t:t + 1] * wx[None, :] + h @ wh, 0.0))
    return h @ wo

ALIGN = {"h%d" % t: ["h%d" % t] for t in range(1, 9)}
