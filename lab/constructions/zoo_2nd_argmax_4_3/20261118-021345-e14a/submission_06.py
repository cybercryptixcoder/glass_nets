
import numpy as np

DESCRIPTION = (
 "A three-step cascade of ReLU threshold detectors. "
 "Step 1: the first input only enters through its sign, h1_j = relu(wx_j*x1) with input-weight "
 "magnitudes 4.7163 and 3.9653 (two units each), so the state is just (relu(x1), relu(-x1)). "
 "Step 2: h2_j = relu(wx_j*x2 + (h1 @ Wh)_j): unit j switches on when x2 beats a drive inherited from "
 "the sign of x1. "
 "Step 3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j): the same detector applied to x3, so unit j is on when x3 "
 "clears a drive inherited from the step-2 switches. Wh is a switch matrix whose entries take only a "
 "handful of magnitudes, each shared by two positions. "
 "Readout: the three class scores are linear in h3 with six magnitudes (16, 18.7, 15.7, 11.45, 1.6, "
 "1.66), so only two score gaps decide: class 0 if score0-score2 is largest, class 1 if score1-score2 "
 "is largest, else class 2. Those gaps live at h3 and are the construction's output.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
s1 = -2.3852   # shared by the two -2.38 entries
s2 = -2.4339
s3 = -1.8940   # shared by the two -1.89 entries
s4 = 1.3398    # shared by the two 1.34 entries
s5 = -2.1996
s6 = 2.6429
s7 = 1.9277
s8 = 2.8057
WH = np.array([[s1, s2, s3, s3],
               [s4, s5, s6, s2],
               [s7, s7, s4, -2.1912],
               [s8, s1, 2.7868, 2.6352]])

# readout magnitudes (six values) and the two decision directions
kA, kB, kC, kD, kE, kF = -15.7403, 15.9993, -18.7243, 11.4769, 1.6175, -1.6650
GA = np.array([kA - kD, kF - kD, kA - kE, kB - kE])
GB = np.array([kB - kD, kC - kD, kC - kE, kF - kE])

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.maximum((p - n)[:, None] * WX, 0.0))
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    h3 = np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0)
    g0 = iv("gap02", h3 @ GA)
    g1 = iv("gap12", h3 @ GB)
    return np.c_[g0, g1, np.zeros_like(g0)]

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"],
         "gap02": ["h3"], "gap12": ["h3"]}
