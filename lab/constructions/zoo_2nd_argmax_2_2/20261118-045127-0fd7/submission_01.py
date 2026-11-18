import numpy as np

DESCRIPTION = (
    "The network compares the two numbers of the sequence: it decides class 1 when x1 > x2 and class 0 when x2 > x1. "
    "Step 1 (h1): the input weight splits x1 by sign, giving the two units the magnitudes of x1: unit0 = 4.174*relu(x1), "
    "unit1 = 6.754*relu(-x1). Step 2 (h2): unit0 fires when x2 exceeds a fraction of x1 - the fraction is 0.382 when x1>0 "
    "and -2.085 when x1<0, i.e. drive0 = x2 - 0.382*relu(x1) + 2.085*relu(-x1); unit1 is the complementary detector, "
    "drive1 = -x2 + 1.141*relu(x1) - 0.752*relu(-x1). Both units use relu. The readout is the difference of the two "
    "detectors: score0 - score1 = 4.174*unit0 - 4.353*6.754*unit1, so class 0 wins exactly when unit0 > 4.353*unit1. "
    "The two drives are the two sides of a comparison, which is why the decision is (up to a tiny bias) the sign of x1-x2."
)

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    p = iv("x1_pos", np.maximum(x1, 0.0))
    m = iv("x1_neg", np.maximum(-x1, 0.0))
    s0 = iv("det_x2_above", np.maximum(x2 - 0.38212 * p + 2.08517 * m, 0.0))
    s1 = iv("det_x2_below", np.maximum(-x2 + 1.14153 * p - 0.75152 * m, 0.0))
    diff = s0 - 4.35334 * s1
    return np.stack([diff, -diff], axis=1)

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"],
         "det_x2_above": ["h2"], "det_x2_below": ["h2"]}
