
import numpy as np

DESCRIPTION = (
 "A three-step cascade of ReLU threshold detectors, one detector block per input. "
 "Step 1: the first input only enters through its sign, h1_j = relu(wx_j*x1) with input weights of "
 "magnitude 4.716 and 3.965 (two units each), so the state is just (relu(x1), relu(-x1)). "
 "Step 2: h2_j = relu(wx_j*x2 + (h1 @ Wh)_j): each unit switches on when x2 beats a drive inherited "
 "from the sign of x1. "
 "Step 3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j): the same detector applied to x3, so unit j is on when "
 "x3 clears a drive inherited from step 2. Wh is a switch matrix built from only eight magnitudes in "
 "a fixed pattern (each appears twice). "
 "Readout: the class scores are linear in h3 with six magnitudes (16, 18.7, 15.7, 11.45, 1.6, 1.66), "
 "so only the two score gaps decide: class 0 if score0-score2 is largest, class 1 if score1-score2 is "
 "largest, else class 2. Those two gaps are the decision variables and live at h3.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
# eight switch magnitudes of the recurrent map
v1 = -2.3852   # also used for the -2.3826 entry
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

# decision directions at h3: (score0 - score2) and (score1 - score2)
GA = [-27.217, -13.102, -17.315, 14.360]
GB = [4.522, -30.155, -20.365, -3.257]
GA = np.array(GA)
GB = np.array(GB)

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.maximum((p - n)[:, None] * WX, 0.0))
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    g0 = iv("gap02", np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0) @ GA)
    g1 = iv("gap12", np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0) @ GB)
    return np.stack([g0, g1, np.zeros(len(x))], 1)

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"],
         "gap02": ["h3"], "gap12": ["h3"]}
