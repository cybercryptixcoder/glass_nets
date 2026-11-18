
import numpy as np
DESCRIPTION = "Placeholder exact-weights construction: recurrence h_t = relu(x_t*Wx + h_{t-1}@Wh), out = h4@Wo. Used to check behaviour/alignment; compression not yet addressed."
Wx = np.array([0.9975, -2.2207, -2.0354, 0.3939])
Wh = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
Wo = np.array([[1.6376, 0.6297, -2.5617, 0.5466],
               [1.0779, -0.534, -1.0086, 0.6374],
               [-1.2079, 1.2053, 0.8859, -0.9309],
               [-2.5583, -0.5449, 3.3309, -0.887]])

def model(x, iv):
    B = x.shape[0]
    h = np.zeros((B, 4))
    for t in range(4):
        h = iv("h%d" % (t+1), np.maximum(x[:, t:t+1]*Wx + h @ Wh, 0))
    return h @ Wo
ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
