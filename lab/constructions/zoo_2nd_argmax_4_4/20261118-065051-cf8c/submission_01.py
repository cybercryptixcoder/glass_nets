import numpy as np

DESCRIPTION = ("Diagnostic exact re-implementation: 4-unit ReLU RNN, h_t = relu(x_t*W_x + h_{t-1}@W_h), "
               "out = h_4 @ W_o. Stage 1: h1 = relu(x1*W_x) lies on one of two rays (sign(x1)).")

Wx = np.array([[0.9975, -2.2207, -2.0354, 0.3939]])
Wh = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
Wo = np.array([[1.6376, 0.6297, -2.5617, 0.5466],
               [1.0779, -0.5340, -1.0086, 0.6374],
               [-1.2079, 1.2053, 0.8859, -0.9309],
               [-2.5583, -0.5449, 3.3309, -0.8870]])

def model(x, iv):
    h = None
    for t in range(4):
        pre = x[:, t:t+1] * Wx
        if h is not None:
            pre = pre + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    return h @ Wo

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
