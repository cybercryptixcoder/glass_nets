
import numpy as np

DESCRIPTION = (
 "Three-step switch cascade. Each hidden unit is a ReLU threshold detector: it turns on when the "
 "current input's drive (wx_j * x_t) beats a drive carried over from the previous state. "
 "Step 1: only the sign of x1 matters, so the state is (relu(x1), relu(-x1)) scaled by the four "
 "input weights (magnitudes 4.716 and 3.965 twice each). "
 "Step 2: h2_j = relu(wx_j*x2 + (h1 @ Wh)_j): unit j switches on when x2 clears a threshold which is "
 "a fixed linear function of |x1| (units 0,2 for x1<0 / x1>0 respectively etc.). "
 "Step 3: h3_j = relu(wx_j*x3 + (h2 @ Wh)_j), same rule with x3. "
 "Readout: the three scores are linear in h3 with only six magnitudes (16, 18.7, 15.7, 11.45, 1.6, 1.66), "
 "so only the two score gaps d02 = s0-s2 and d12 = s1-s2 matter: class 0 if d02 is largest, "
 "class 1 if d12 is largest, else class 2. Those two gaps are the whole decision and live at h3.")

WX = np.array([-4.7163, 3.9653, 4.6995, -3.9652])
# recurrent switch map, four rows (rows = source unit, columns = target unit)
R0 = [-2.3877, -2.4339, -1.8931, -1.8948]
R1 = [1.3387, -2.1996, 2.6429, -2.4339]
R2 = [1.9277, 1.9331, 1.3409, -2.1912]
R3 = [2.8057, -2.3826, 2.7868, 2.6352]
WH = np.array([R0, R1, R2, R3])

# readout magnitudes (six distinct values, reused across the three columns)
kA = -15.72   # large negative on units 0,2 for class 0
kB = 16.0     # large positive, the "unit 0 / unit 3" signal
kC = -18.73   # large negative on units 1,2 for class 1
kD = 11.45    # positive on units 0,1 for class 2
kE = 1.61     # positive on units 2,3 for class 2
kF = -1.665   # small negative
C0 = np.array([kA, kF, kA, kB])
C1 = np.array([kB, kC, kC, kF])
C2 = np.array([kD, kD, kE, kE])
# only the two score gaps matter for the decision
G0 = C0 - C2
G1 = C1 - C2

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.stack([p, n], 1) @ np.array([[0.0, 3.9653, 4.6995, 0.0],
                                                  [4.7163, 0.0, 0.0, 3.9652]]))
    h2 = iv("h2", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    h3 = iv("h3", np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0))
    g0 = iv("gap02", h3 @ G0)
    g1 = iv("gap12", h3 @ G1)
    return np.stack([g0, g1, np.zeros(len(x))], 1)

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"], "h3": ["h3"],
         "gap02": ["h3"], "gap12": ["h3"]}
