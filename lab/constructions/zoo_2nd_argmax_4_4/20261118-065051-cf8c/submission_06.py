import numpy as np

DESCRIPTION = ("4-unit ReLU RNN, h_t = relu(x_t*W_x + h_{t-1}@W_h), out = h_4 @ W_o. "
               "h1 is a pure gate: the sign of x1 picks one of two rays, "
               "h1 = relu(x1)*relu(W_x) + relu(-x1)*relu(-W_x), so site h1 is 2-dimensional. "
               "Steps 2-4 are the dense recurrence; only the three margins "
               "m_k = h4 . (W_o[:,k+1]-W_o[:,0]) matter for the decision, so class 0 is scored 0. "
               "Still a diagnostic re-implementation of the weights.")

Wx = np.array([0.9975, -2.2207, -2.0354, 0.3939])
Wh = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
D  = np.array([[-1.008, -4.199, -1.091],
               [-1.612, -2.087, -0.441],
               [2.413, 2.094, 0.277],
               [2.013, 5.889, 1.671]])

A1 = np.maximum(Wx, 0)
A2 = np.maximum(-Wx, 0)

def model(x, iv):
    u = iv("xp", np.maximum(x[:, 0], 0.0))
    v = iv("xn", np.maximum(-x[:, 0], 0.0))
    h = u[:, None] * A1 + v[:, None] * A2
    for t in range(1, 4):
        pre = x[:, t:t+1] * Wx + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    m = iv("m", h @ D)
    return np.column_stack([np.zeros(len(h)), m])

ALIGN = {"xp": ["h1"], "xn": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "m": ["h4"]}
