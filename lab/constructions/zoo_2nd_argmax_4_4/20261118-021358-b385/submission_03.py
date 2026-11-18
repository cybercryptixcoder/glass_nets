
import numpy as np
DESCRIPTION = """Step 1 (h1): the first hidden state depends only on x1. The input weights have two signs, so units
{0,2} carry a fixed multiple of relu(-x1) (the negative part of x1) and units {1,3} a fixed multiple of
relu(x1) (the positive part). Two variables rp = relu(x1) and rn = relu(-x1) therefore span h1 exactly.
Steps 2-4: each step adds the new input's drive (the same four input gains g, one per unit) to a fixed
recurrent mixing of the previous state and rectifies. The output is a fixed linear readout of the final state.
The whole sequence is thus a rectified linear filter of the input stream in which the earlier inputs dominate."""

g  = np.array([-2.0354, 0.9975, -2.2207, 0.3939])
Wh = np.array([[-0.1426, 0.5307, 0.9157, -0.0335],
               [-0.2137, 2.4333, 2.5231, 1.3203],
               [ 2.8348, 2.4935, -0.7525, -2.7988],
               [-0.4893, 0.4774, -1.4067, -2.1697]])
Wo = np.array([[-2.5617, -0.9309, 3.3309, -0.5340],
               [-2.5583, 1.0779, 0.6297, 1.6376],
               [-1.2079, 0.8859, -0.5449, -1.0086],
               [ 0.6374, 1.2053, 0.5466, -0.8870]])

def model(x, iv):
    rp = iv("rp", np.maximum(x[:, 0], 0.0))
    rn = iv("rn", np.maximum(-x[:, 0], 0.0))
    h = np.stack([g[0] * rn, g[1] * rp, g[2] * rn, g[3] * rp], axis=1)
    for t in range(1, 4):
        h = iv("s%d" % (t + 1), np.maximum(x[:, [t]] * g + h @ Wh, 0.0))
    return h @ Wo

ALIGN = {"rp": ["h1"], "rn": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"]}
