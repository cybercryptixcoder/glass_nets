
import numpy as np

DESCRIPTION = """
Two-step ReLU recurrence, all weights rank-1-ish. Step 1 splits the sign of input 1 into two
nonnegative parts: p = relu(x1) drives hidden unit 1 (positive weight) and n = relu(-x1) drives
hidden unit 2; so h1 = [a*p, b*n]. At step 2 each hidden unit is a ReLU of (input 2) plus a fixed
linear combination of p and n, rescaled so the coefficient of x2 is 1:
  u = relu(x2 - 0.3821*p + 2.0852*n)   = h2_1 / 4.174
  w = relu(-x2 + 1.1415*p - 0.7515*n)  = h2_2 / 6.754
The readout compares the two units with opposite signs (a difference of the two class scores), so
the decision is  class 0  iff  u > 4.3533 * w.  Returned as score (s, 0).
"""

C1 = -0.38212355971336365
C2 = 2.0851662362478636
C3 = 1.1415294349709517
C4 = -0.7515226602554321
K  = 4.353341987440024

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    p = iv("p", np.maximum(x1, 0.0))       # = h1_1 / 4.174
    n = iv("n", np.maximum(-x1, 0.0))      # = h1_2 / 6.754
    u = iv("u", np.maximum(x2 + C1 * p + C2 * n, 0.0))    # = h2_1 / 4.174
    w = iv("w", np.maximum(-x2 + C3 * p + C4 * n, 0.0))   # = h2_2 / 6.754
    s = u - K * w
    return np.stack([s, np.zeros_like(s)], axis=1)

ALIGN = {"p": ["h1"], "n": ["h1"], "u": ["h2"], "w": ["h2"]}
