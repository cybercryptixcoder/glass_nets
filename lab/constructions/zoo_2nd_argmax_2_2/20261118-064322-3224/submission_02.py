
import numpy as np

DESCRIPTION = """
The network is a two-step ReLU comparator: it decides class 0 exactly when input 2 is larger
than input 1 (out0-out1 > 0  <=>  x2 > x1); ties at x2 = x1 are float32 accidents.

Stage 1 (h1): the input x1 is split by sign into two nonnegative parts, p = relu(x1) and
n = relu(-x1); hidden unit 1 holds a p and unit 2 holds b n (a > 0 > b). So the state keeps
"how positive x1 is" and "how negative x1 is" in separate units.

Stage 2 (h2): each hidden unit is one ReLU "vote" about x2:
  unit 1: u = relu(x2 + C1 p + C2 n)  > 0 when x2 is above a shallow line through x1 (slope -C1),
  unit 2: w = relu(-x2 + C3 p + C4 n) > 0 when x2 is below a steeper line through x1,
and both lines cross zero at x2 = x1, so the two votes overlap and are calibrated (below).

Output: the readout differences the two units with opposite sign, i.e. score = u - K w.
Because both ReLU thresholds pass through x2 = x1 and the readout calibrates the two slopes
(K = 1.0/... see below), the crossover u = K w falls exactly on x2 = x1: the score is a
re-scaled version of x2 - x1, and "largest output" is the comparison x2 > x1.
"""
# --- three numbers: the two slopes of the upper vote and the readout balance -------------
C3 = 1.1415294349709517     # slope of the lower vote w on the positive part p
C4 = -0.7515226602554321    # slope of the lower vote w on the negative part n
K  = 4.365709793255747      # readout balance between the two votes

# the upper vote's slopes follow from calibration: both vote thresholds pass through x2 = x1
# (C3 - 1)/K = 1 + C1  and  C2 = 1 + K (1 + C4)
C1 = K * C3 - (1.0 + K)
C2 = 1.0 + K * (1.0 + C4)

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    p = iv("p", np.maximum(x1, 0.0))        # = h1_1 / a  (x1 positive part)
    n = iv("n", np.maximum(-x1, 0.0))       # = h1_2 / b  (x1 negative part)
    u = iv("u", np.maximum(x2 + C1 * p + C2 * n, 0.0))    # = h2_1 / a, upper vote
    w = iv("w", np.maximum(-x2 + C3 * p + C4 * n, 0.0))   # = h2_2 / b, lower vote
    s = u - K * w
    return np.stack([s, np.zeros_like(s)], axis=1)

ALIGN = {"p": ["h1"], "n": ["h1"], "u": ["h2"], "w": ["h2"]}
