
import numpy as np

DESCRIPTION = (
 "A three-step cascade of ReLU threshold detectors, with a three-way comparison readout. "
 "Step 1: the first input only enters through its sign, h1_j = relu(wx_j*x1) with input-weight "
 "magnitudes 4.7163 and 3.9653 (two units each), so the whole state is (relu(x1), relu(-x1)). "
 "Step 2: h2_j = relu(wx_j*x2 + (h1 @ Wh)_j): unit j switches on when x2 beats a drive inherited from "
 "the sign of x1. "
 "Step 3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j): the same detector applied to x3, so unit j is on when x3 "
 "clears a drive inherited from the step-2 switches; Wh's entries take only a handful of magnitudes, "
 "each shared by two positions. "
 "Readout: the three class scores are the sum of the h3 units weighted by three columns of a small "
 "table (magnitudes 16, 18.7, 15.7, 11.45, 1.6, 1.66); class 0 wants unit 3 on, class 1 wants unit 0 "
 "on, class 2 wants units 1 and 2 on. The scores themselves are the decision and live at h3.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
s1 = -2.3852
s2 = -2.4339
s3 = -1.8940
s4 = 1.3398
s5 = -2.1996
s6 = 2.6429
s7 = 1.9277
s8 = 2.8057
WH = np.array([[s1, s2, s3, s3],
               [s4, s5, s6, s2],
               [s7, s7, s4, -2.1912],
               [s8, s1, 2.7868, 2.6352]])

kA, kB, kC, kD = -15.7403, 15.9993, -18.7243, 11.4769
kE, kF, kG = 11.4305, 1.6175, 1.5993
kH, kI, kJ, kK = -1.6718, -1.6575, -15.6973, 15.9592
C0 = np.array([kA, kH, kJ, kK])
C1 = np.array([kB, kC, -18.7477, kI])
C2 = np.array([kD, kE, kF, kG])

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.maximum((p - n)[:, None] * WX, 0.0))
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    h3 = np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0)
    s0 = iv("score0", h3 @ C0)
    s1c = iv("score1", h3 @ C1)
    s2c = iv("score2", h3 @ C2)
    return np.c_[s0, s1c, s2c]

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"],
         "score0": ["h3"], "score1": ["h3"], "score2": ["h3"]}
