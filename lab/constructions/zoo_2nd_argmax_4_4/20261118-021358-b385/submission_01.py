
import numpy as np
DESCRIPTION = """Stage 1 (h1): h1_j = relu(W_x_j * x1), so the first hidden state depends only on the sign of x1:
units 0,2 hold relu(-x1) scaled, units 1,3 hold relu(x1) scaled. Two variables rp=relu(x1), rn=relu(-x1)
span h1. Later stages are being investigated; for now the recurrent updates are carried with measured constants."""

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
    h = iv("s1", h)
    out = h
    for t in range(1, 4):
        h = iv("s%d" % (t + 1), np.maximum(x[:, [t]] * Wx + h @ Wh, 0.0))
        out = h
    return out @ Wo

ALIGN = {"rp": ["h1"], "rn": ["h1"],
         "s1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"]}
