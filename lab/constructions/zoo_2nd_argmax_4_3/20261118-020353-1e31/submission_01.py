
import numpy as np

DESCRIPTION = (
 "The network scores the three sequence positions and picks the largest; the class it picks is the "
 "position of the second-largest (median) input.  Stage 1: the four units of h1 are all proportional to "
 "relu(+x1) or relu(-x1), i.e. h1 holds the signed pair (x1)^+, (x1)^- with redundant scalings, so x1 = a1-b1. "
 "Stage 2: h2 mixes the retained x1 with the new x2, so its units are hinges of (x1,x2); they are well "
 "described by the signed pairs of both inputs together with the pairwise relu's relu(x1-x2), relu(x2-x1), "
 "relu(x1+x2), relu(-x1-x2).  Final readout: the median position is the position whose total distance to "
 "the other two values is smallest, score_i = -(|x_i-x_j| + |x_i-x_k|), and |.| splits into relu's; the "
 "network's argmax over its three output scores is that position."
)

def model(x, iv):
    x1c = x[:, 0]; x2c = x[:, 1]; x3 = x[:, 2]
    a1 = iv("a1", np.maximum(x1c, 0.0))
    b1 = iv("b1", np.maximum(-x1c, 0.0))
    x1 = a1 - b1
    a2 = iv("a2", np.maximum(x2c, 0.0))
    b2 = iv("b2", np.maximum(-x2c, 0.0))
    x2 = a2 - b2
    d = x1 - x2
    r12p = iv("r12p", np.maximum(d, 0.0))
    r12m = iv("r12m", np.maximum(-d, 0.0))
    s = x1 + x2
    r12s = iv("r12s", np.maximum(s, 0.0))
    r12n = iv("r12n", np.maximum(-s, 0.0))
    # keep the variables alive in a neutral (affine) way for the readout
    x1 = a1 - b1 + 0.0 * (r12p + r12m + r12s + r12n)
    x2 = a2 - b2 + 0.0 * (r12p + r12m + r12s + r12n)
    s1 = -(np.abs(x1 - x2) + np.abs(x1 - x3))
    s2 = -(np.abs(x2 - x1) + np.abs(x2 - x3))
    s3 = -(np.abs(x3 - x1) + np.abs(x3 - x2))
    return 2.0 * np.stack([s1, s2, s3], axis=1)

ALIGN = {"a1": ["h1"], "b1": ["h1"],
         "a2": ["h2"], "b2": ["h2"],
         "r12p": ["h2"], "r12m": ["h2"], "r12s": ["h2"], "r12n": ["h2"]}
