import numpy as np

DESCRIPTION = ("4-unit ReLU RNN, h_t = relu(x_t*W_x + h_{t-1}@W_h). Decisions are the argmax of h_4 @ W_o, "
               "and since adding the same vector to every class column leaves all comparisons unchanged, the "
               "readout only needs the three margins m_i = h_4.(c_i - c_0): class 0 plays the role of the "
               "zero reference, so out = [0, m1, m2, m3] (12 numbers instead of 16). "
               "Step 1 is a pure gate: sign(x1) selects one of two rays, h1 = relu(x1)*relu(W_x) + relu(-x1)*relu(-W_x). "
               "Steps 2-4 are the exact dense recurrence.")

Wx = np.array([0.9975, -2.2207, -2.0354, 0.3939])
Wh = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
# margins of each class against class 0: m_i = h4 . (column_i - column_0)
D = np.array([[-0.5597, -2.8455, -4.1959],
              [-1.1637, 0.5756, -1.1746],
              [1.5531, 3.4476, 5.8926],
              [0.0908, -1.4775, -1.4336]])

A1 = np.maximum(Wx, 0)
A2 = np.maximum(-Wx, 0)

def model(x, iv):
    u = iv("xp", np.maximum(x[:, 0], 0.0))
    v = iv("xn", np.maximum(-x[:, 0], 0.0))
    h = u[:, None] * A1 + v[:, None] * A2
    for t in range(1, 4):
        pre = x[:, t:t+1] * Wx + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    m = iv("margins", h @ D)
    return np.column_stack([np.zeros(len(x)), m[:, 0], m[:, 1], m[:, 2]])

ALIGN = {"xp": ["h1"], "xn": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
