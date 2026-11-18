
import numpy as np

DESCRIPTION = (
 "Three-step switch cascade of threshold detectors. "
 "Step 1: the first input only enters through its sign, h1_j = relu(wx_j*x1) with input weights "
 "(4.7163, 3.9653, 4.6995, 3.9652) in magnitude, so the whole state is the pair "
 "x1pos = relu(x1), x1neg = relu(-x1). "
 "Step 2: h2_j = relu(wx_j*x2 + (h1 @ Wh)_j): each unit switches on when x2 clears a threshold that "
 "is a fixed linear function of the previous state. "
 "Step 3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j): the same rule applied to x3, so unit j is on when x3 "
 "beats a drive inherited from step 2. "
 "Readout: the three class scores are linear in h3 with only six magnitudes (16, 18.7, 15.7, 11.45, "
 "1.6, 1.66), so only the two score gaps matter: class 0 when h3 @ (c0-c2) is largest, class 1 when "
 "h3 @ (c1-c2) is largest, otherwise class 2.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
R0 = [-2.3877, -2.4339, -1.8931, -1.8948]
R1 = [1.3387, -2.1996, 2.6429, -2.4339]
R2 = [1.9277, 1.9331, 1.3409, -2.1912]
R3 = [2.8057, -2.3826, 2.7868, 2.6352]
WH = np.array([R0, R1, R2, R3])

# decision directions at h3: score(class0)-score(class2) and score(class1)-score(class2)
GA = [-27.17, -13.12, -17.33, 14.39]
GB = [4.55, -30.18, -20.34, -3.28]
GA = np.array(GA); GB = np.array(GB)

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.maximum((p - n)[:, None] * WX, 0.0))
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    h3 = iv("h3", np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0))
    g0 = h3 @ GA
    g1 = h3 @ GB
    return np.stack([g0, g1, np.zeros(len(x))], 1)

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"], "h3": ["h3"]}
