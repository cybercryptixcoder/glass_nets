
import numpy as np

DESCRIPTION = (
 "The network is a three-step switch cascade: each hidden unit is a ReLU threshold that turns on "
 "when the current input beats a drive carried over from the previous state. "
 "Step 1: only the sign of x1 matters, h1 = (relu(x1), relu(-x1)) scaled by the input weights "
 "3.965/4.700 for x1>0 (units 1,2) and 4.716/3.965 for x1<0 (units 0,3). "
 "Step 2: h2_j = relu(wx_j*x2 + drive_j) with drive = (relu(x1), relu(-x1)) @ A, i.e. each unit is on "
 "when x2 clears a threshold that is a fixed linear function of |x1| (or of x1's sign). "
 "Step 3: the same again with x3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j). "
 "Readout: the three scores compare which h3 units dominate; the coefficient table has only six "
 "distinct magnitudes -15.72, 16.00, -18.73, 11.45, 1.61, -1.665 - so class 0 = 'unit 3 big', "
 "class 1 = 'unit 0 big', class 2 = 'units 1,2 big'.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
M1 = np.array([[0.0, 3.9653, 4.6995, 0.0],
               [4.7163, 0.0, 0.0, 3.9652]])
# drive of each h2 unit per unit of relu(x1) and relu(-x1)
A = np.array([[14.368, 0.363, 16.781, -19.949],
              [-0.136, -20.927, 2.122, 1.513]])
# recurrent map into h3 (rows = h2 unit, eight distinct values)
WH = np.array([[-2.385, -2.434, -1.894, -1.894],
               [1.340, -2.195, 2.639, -2.434],
               [1.930, 1.930, 1.340, -2.195],
               [2.796, -2.385, 2.796, 2.639]])
# readout, six distinct magnitudes
WO = np.array([[-15.72, 16.00, 11.45],
               [-1.665, -18.73, 11.45],
               [-15.72, -18.73, 1.61],
               [16.00, -1.665, 1.61]])

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.stack([p, n], 1) @ M1)
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + np.stack([p, n], 1) @ A, 0.0))
    h3 = iv("h3", np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0))
    return h3 @ WO

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"], "h3": ["h3"]}
