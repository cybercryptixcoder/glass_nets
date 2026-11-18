
import numpy as np

DESCRIPTION = ("Mechanism: a plain ReLU recurrence. At each step the network forms the pre-activation "
               "z_t = x_t * W_x + h_{t-1} W_h (no bias), passes it through ReLU, and after the 8th step "
               "reads 8 class scores as out = h_8 @ W_o; the decision is the largest score. There is no "
               "hidden symbolic rule: the 8 units are generic growing ReLU features (the recurrent matrix has "
               "spectral radius 1.61, so activations grow to order 30 by step 8 and several units switch "
               "on/off), and the class logits are a fixed linear readout of that state. Only two directions "
               "are present at step 1 (positive and negative part of the first input); later states fill the "
               "whole 8-dim space. The weights are stored here as integers over a common scale.")

Wx = np.array([-207, 30, -1203, -151, -467, 3, 619, 4]) / 10**3
Wh = np.array([[82, 387, 740, -1331, -646, 143, -323, 688],
               [-138, 480, -837, 94, 773, -1200, -449, 1909],
               [668, -155, 235, 36, 901, 1754, 966, -520],
               [1227, 1953, -584, -671, -194, 970, 168, 114],
               [-318, -865, -451, 1279, 272, -43, 537, -409],
               [-214, -23, -318, 1098, -663, 785, 866, 614],
               [1400, -115, 513, 330, 1208, 588, 488, -1103],
               [-246, 791, 33, 104, -245, 282, 1816, -600]]) / 10**3
Wo = np.array([[-852, -352, -243, -179, -558, -129, 2122, -372],
               [679, 115, -50, -223, -1256, -216, 363, 126],
               [-160, 72, -136, 155, 534, 167, 659, -4450],
               [424, 23, 156, -199, -288, -2593, 1314, 45],
               [557, -3, 212, 173, 481, 865, -2247, 268],
               [534, 334, 556, 275, -111, 244, -2480, 344],
               [-433, -99, -448, 105, 707, 108, 27, -96],
               [-4, 11, 67, -658, 372, 29, -364, 13]]) / 10**3

def model(x, iv):
    B = x.shape[0]
    h = np.zeros((B, 8))
    for t in range(8):
        z = np.outer(x[:, t], Wx) + h @ Wh
        h = iv("h%d" % (t + 1), np.maximum(z, 0.0))
    return h @ Wo

ALIGN = {"h%d" % t: ["h%d" % t] for t in range(1, 9)}
