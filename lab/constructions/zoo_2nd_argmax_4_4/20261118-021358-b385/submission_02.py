
import numpy as np
DESCRIPTION = """Step 1: the first hidden state depends only on x1: units {0,2} carry a negative copy of x1 and units
{1,3} a positive one, i.e. h1 = [c0*relu(-x1), c1*relu(x1), c2*relu(-x1), c3*relu(x1)] (a sign split of x1).
So two variables rp = relu(x1), rn = relu(-x1) span h1. Each later step mixes the new input with the previous
state through a fixed recurrent coupling (measured constants), rectified; the output reads the final state."""

Wx = np.array([[-2.0354, 0.9975, -2.2207, 0.3939]])
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
    h = np.stack([2.0354 * rn, 0.9975 * rp, 2.2207 * rn, 0.3939 * rp], axis=1)
    for t in range(1, 4):
        h = iv("s%d" % (t + 1), np.maximum(x[:, [t]] * Wx + h @ Wh, 0.0))
    return h @ Wo

ALIGN = {"rp": ["h1"], "rn": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"]}
