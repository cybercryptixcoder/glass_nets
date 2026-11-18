import numpy as np

DESCRIPTION = ("4-unit ReLU RNN rescaled so the input weight is the sign pattern (+1,-1,-1,+1): "
               "with unit scales s=|W_x| the recurrence keeps its form and relu still acts unitwise, so in the "
               "rescaled state u_t = h_t/s: u_t = relu(x_t*(+1,-1,-1,+1) + u_{t-1} @ M). Each unit thus receives "
               "+-x_t directly; all coupling is the 4x4 matrix M. Step 1 is a pure gate (sign of x1 picks one of two "
               "rays), so site h1 is 2-dimensional. Steps 2-4 are the dense recurrence. The decision only needs the "
               "three margins m_k = u_4 . (W_o[:,k+1]-W_o[:,0])*s, so class 0 is scored 0. M and the margins are "
               "fitted jointly (the margins absorb the residual), which is why agreement is 99.8% rather than 100%.")

M = np.array([[-0.212043, 1.271137, -0.071773, 2.302952],
              [ 5.488086, 1.320625, -0.530692, 14.266274],
              [-4.5,     -1.289620,  0.527514, -14.5],
              [-0.012784, 0.084456,  0.484783, -0.75]])
D = np.array([[-0.205007, -0.855236, -0.222272],
              [-0.728108, -0.944793, -0.199427],
              [ 1.0,       0.869319,  0.115109],
              [ 0.161895,  0.476049,  0.135044]])
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
