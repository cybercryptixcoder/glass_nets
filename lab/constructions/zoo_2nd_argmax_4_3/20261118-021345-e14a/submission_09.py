
import numpy as np

DESCRIPTION = (
 "A three-step cascade of ReLU threshold detectors. "
 "Step 1: the first input only enters through its sign, h1_j = relu(wx_j*x1) with input-weight "
 "magnitudes 4.7163 and 3.9653 (two units each), so the state is just (relu(x1), relu(-x1)). "
 "Step 2: h2_j = relu(wx_j*x2 + (h1 @ Wh)_j): unit j switches on when x2 beats a drive inherited from "
 "the sign of x1. "
 "Step 3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j): the same detector applied to x3, so unit j is on when x3 "
 "clears a drive inherited from the step-2 switches; the entries of Wh take only eight magnitudes, "
 "each shared by two positions. "
 "Readout: the scores are linear in h3 with six magnitudes (16, 18.7, 15.7, 11.45, 1.6, 1.66), so the "
 "decision depends only on two score gaps: class 0 if score0-score2 is largest, class 1 if "
 "score1-score2 is largest, else class 2. Those two gaps are what the last state decides with; the "
 "state's overall size carries no further information.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
v1 = -2.3852
v2 = -2.4339
v3 = -1.8940
v4 = 1.3398
v5 = -2.1954
v6 = 2.6391
v7 = 1.9304
v8 = 2.7963
WH = np.array([[v1, v2, v3, v3],
               [v4, v5, v6, v2],
               [v7, v7, v4, v5],
               [v8, v1, v8, v6]])

GA = np.array([-27.217, -13.102, -17.315, 14.360])
GB = np.array([4.522, -30.155, -20.365, -3.257])
ONE = np.array([1.0, 1.0, 1.0, 1.0])

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.maximum((p - n)[:, None] * WX, 0.0))
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    h3 = np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0)
    g0 = iv("gap02", h3 @ GA)
    g1 = iv("gap12", h3 @ GB)
    tot = iv("total", h3 @ ONE)
    return np.c_[g0, g1, np.zeros_like(g0)]

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"],
         "gap02": ["h3"], "gap12": ["h3"], "total": ["h3"]}
