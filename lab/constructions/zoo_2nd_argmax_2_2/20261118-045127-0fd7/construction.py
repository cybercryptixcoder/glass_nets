import numpy as np

DESCRIPTION = (
    "The network compares the two numbers of the sequence: it decides class 1 when x1 > x2, class 0 when x2 > x1. "
    "Step 1 (h1): the input weight splits x1 by sign, so the two units carry the magnitude of x1: "
    "unit0 = 4.174*relu(x1), unit1 = 6.754*relu(-x1) (i.e. relu(x1) and relu(-x1) up to fixed gains, which the "
    "variables absorb). "
    "Step 2 (h2): the two units are the two sides of a comparison. Unit0 fires when x2 is above a line through the "
    "origin, unit1 when x2 is below another line: with C = 0.38212 and Q = 2.08517 the lines for positive x1 are "
    "x2 = C*x1 (unit0) and x2 = S*x1 (unit1), and for negative x1 they are x2 = -Q*x1 (unit0) and x2 = -T*x1 (unit1). "
    "So drive0 = x2 - C*relu(x1) + Q*relu(-x1), drive1 = -x2 + S*relu(x1) - T*relu(-x1), each passed through relu. "
    "The network is an (almost) exact comparator, so the two lines are mirror images about the diagonal x2 = x1: "
    "this fixes S and T once the readout weight R is known, S = 1 + (1-C)/R and T = 1 - (Q-1)/R, with only C, Q, R "
    "left as free numbers. "
    "Readout: the two output rows are nearly antisymmetric, so the score difference is unit0 - R*unit1 (in the "
    "normalised units of the drives); class 0 wins when drive0 > R*drive1, which is exactly x2 > x1."
)

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    R = 4.35334
    C = 0.38212
    Q = 2.08517
    S = 1.0 + (1.0 - C) / R
    T = 1.0 - (Q - 1.0) / R
    p = iv("x1_pos", np.maximum(x1, 0.0))
    m = iv("x1_neg", np.maximum(-x1, 0.0))
    d0 = iv("x2_above", np.maximum(x2 - C * p + Q * m, 0.0))
    d1 = iv("x2_below", np.maximum(-x2 + S * p - T * m, 0.0))
    diff = d0 - R * d1
    return np.stack([diff, -diff], axis=1)

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"],
         "x2_above": ["h2"], "x2_below": ["h2"]}
