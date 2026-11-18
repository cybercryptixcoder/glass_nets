import numpy as np

DESCRIPTION = (
    "Decision: class 0 iff x2 > x1, class 1 iff x2 < x1 (the net compares the two inputs). "
    "Step 1 splits x1 by sign: h1 = [a*relu(x1), |b|*relu(-x1)] (variables p, n). "
    "Step 2 rectifies two combinations: w0 = relu(x2 - c0*p + cn*n) (h2_0 = a*w0), "
    "w1 = relu(-x2 + d0*p - dn*n) (h2_1 = |b|*w1). They are proportional to the two distances "
    "x2-above-x1 and x1-above-x2 measured along the two branches; the readout is rank-1 and "
    "class 0 wins iff w0 > R*w1. The slopes c0, cn follow from requiring the boundary to be the "
    "line x2=x1 given the two measured slopes d0, dn."
)

R = 4.35335
d0 = 1.14152943
dn = 0.75152266
c0 = 1.0 + (1.0 - d0) * R
cn = 1.0 + (1.0 - dn) * R

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    p = iv("p", np.maximum(x1, 0.0))
    n = iv("n", np.maximum(-x1, 0.0))
    w0 = iv("w0", np.maximum(x2 - c0 * p + cn * n, 0.0))
    w1 = iv("w1", np.maximum(-x2 + d0 * p - dn * n, 0.0))
    s = w0 - R * w1
    return np.stack([s, -s], axis=1)

ALIGN = {"p": ["h1"], "n": ["h1"], "w0": ["h2"], "w1": ["h2"]}
