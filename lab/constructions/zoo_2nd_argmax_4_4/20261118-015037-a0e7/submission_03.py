
import numpy as np
DESCRIPTION = ("The state is four rectified accumulators. Step 1 gates the first input by sign: units 0 and 3 take "
               "+x1, units 1 and 2 take -x1. Every later step adds x_t (times a fixed input gain per unit) to the "
               "previous state, mixed by a dense recurrent gain matrix, and rectifies: h_t = relu(x_t*Wx + h_{t-1}@Wh). "
               "The gains are all sub-critical per step but the recurrent matrix amplifies (largest eigenvalue 2.56), "
               "so activity grows over the four steps. The output scores each class by projecting the final state on a "
               "fixed direction; only the three contrasts against class 0 matter, so out = [0, (w1-w0).h4, (w2-w0).h4, "
               "(w3-w0).h4] and the decision is argmax over those four contrasts.")
WX = np.array([0.9975, -2.2207, -2.0354, 0.3939])
WH = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
DW = np.array([[-0.5597, -1.1637, 1.5531, 0.0908],   # class1 - class0 direction
               [-2.8455, 0.5756, 3.4476, -1.4775],   # class2 - class0
               [-4.1959, -1.1746, 5.8926, -1.4336]])  # class3 - class0

def model(x, iv):
    h = np.zeros_like(x)
    for t in range(4):
        drive = x[:, t:t+1] * WX[None, :] + h @ WH
        h = iv("state%d" % (t + 1), np.maximum(drive, 0.0))
    contrasts = h @ DW.T
    return np.hstack([np.zeros_like(contrasts[:, :1]), contrasts])

ALIGN = {"state1": ["h1"], "state2": ["h2"], "state3": ["h3"], "state4": ["h4"]}
