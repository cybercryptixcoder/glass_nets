import numpy as np

DESCRIPTION = ("4-unit ReLU RNN rescaled so that the input weight becomes the sign pattern (+1,-1,-1,+1): "
               "with unit scales s=|W_x| the recurrence keeps its form with relu, so in the rescaled state "
               "u_t = h_t/s we have u_t = relu(x_t*(+1,-1,-1,+1) + u_{t-1} @ M) with a single 4x4 matrix M. "
               "So each unit receives +-x_t directly and the only coupling is M. "
               "Step 1 is a pure gate: the sign of x1 selects one of two rays (units 1,2 or units 0,3), "
               "so site h1 is 2-dimensional. Step 2-4 are the dense recurrence. "
               "Only the three margins m_k = u_4 . (W_o[:,k+1]-W_o[:,0]) * s matter, so class 0 scores 0.")

M = np.array([[-0.21374, 1.27334, -0.06990, 2.31898],
              [ 5.41727, 1.32034, -0.53387, 14.22469],
              [-4.42718, -1.28935, 0.53066, -14.46224],
              [-0.01322, 0.08467, 0.48256, -0.75254]])
D = np.array([[-1.00531, -4.18870, -1.08818],
              [-3.57938, -4.63342, -0.97817],
              [ 4.91167,  4.26159,  0.56386],
              [ 0.79306,  2.31971,  0.65831]])
E1 = np.array([1.0, 0.0, 0.0, 1.0])
E2 = np.array([0.0, 1.0, 1.0, 0.0])
SGN = np.array([1.0, -1.0, -1.0, 1.0])

def model(x, iv):
    u = iv("xp", np.maximum(x[:, 0], 0.0))
    v = iv("xn", np.maximum(-x[:, 0], 0.0))
    h = u[:, None] * E1 + v[:, None] * E2
    for t in range(1, 4):
        pre = x[:, t:t+1] * SGN + h @ M
        h = iv("u%d" % (t+1), np.maximum(pre, 0.0))
    m = h @ D
    return np.column_stack([np.zeros(len(h)), m])

ALIGN = {"xp": ["h1"], "xn": ["h1"], "u2": ["h2"], "u3": ["h3"], "u4": ["h4"]}
