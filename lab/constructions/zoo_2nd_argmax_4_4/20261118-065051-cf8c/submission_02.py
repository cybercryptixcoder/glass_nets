import numpy as np

DESCRIPTION = ("4-unit ReLU RNN, h_t = relu(x_t*W_x + h_{t-1}@W_h), out = h_4 @ W_o. "
               "Step 1: h1 lives on one of two rays: h1 = u*a + v*b with u=relu(x1), v=relu(-x1), "
               "a=(1.0,0,0,0.394), b=(0,2.221,2.035,0). Later steps still exact (diagnostic).")

Wx = np.array([0.9975, -2.2207, -2.0354, 0.3939])
a = np.array([0.9975, 0.0, 0.0, 0.3939])
b = np.array([0.0, 2.2207, 2.0354, 0.0])
Wh = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
Wo = np.array([[1.6376, 0.6297, -2.5617, 0.5466],
               [1.0779, -0.5340, -1.0086, 0.6374],
               [-1.2079, 1.2053, 0.8859, -0.9309],
               [-2.5583, -0.5449, 3.3309, -0.8870]])

def model(x, iv):
    u = iv("xp", np.maximum(x[:, 0], 0.0))
    v = iv("xn", np.maximum(-x[:, 0], 0.0))
    h = iv("h1", u[:, None] * a + v[:, None] * b)
    for t in range(1, 4):
        pre = x[:, t:t+1] * Wx + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    return h @ Wo

ALIGN = {"xp": ["h1"], "xn": ["h1"], "h1": ["h2"], "h2": ["h3"], "h3": ["h4"], "h4": ["h4"]}
