import numpy as np

DESCRIPTION = ("4-unit ReLU RNN rescaled so the input weight is the sign pattern (+1,-1,-1,+1): with unit scales "
               "s=|W_x| the recurrence form and unitwise relu survive, so in the rescaled state u_t = h_t/s we have "
               "u_t = relu(x_t*(+1,-1,-1,+1) + u_{t-1} @ M). Each unit thus receives +-x_t directly and all coupling "
               "is the 4x4 matrix M. Step 1 is a pure gate (the sign of x1 picks one of two rays), so site h1 is "
               "2-dimensional; steps 2-4 are the dense recurrence. Only the three margins "
               "m_k = u_4 . (W_o[:,k+1]-W_o[:,0])*s matter, so class 0 scores 0. M and the margins are fitted jointly "
               "with four entries of M held at round values (0.5, -0.5, 14), so agreement is 98.9% not 100%.")

M = np.array([[-0.221580, 1.265520, -0.083989, 2.202613],
              [ 5.533748, 1.321555, -0.5,      14.0],
              [-4.545520, -1.287757, 0.5,     -14.215864],
              [-0.007837, 0.092553,  0.5,      -0.725877]])
D = np.array([[-0.207069, -0.868111, -0.225856],
              [-0.724419, -0.936870, -0.197053],
              [ 1.0,       0.863421,  0.112810],
              [ 0.164933,  0.490611,  0.139345]])
E1 = np.array([1.0, 0.0, 0.0, 1.0])
E2 = np.array([0.0, 1.0, 1.0, 0.0])
SGN = np.array([1.0, -1.0, -1.0, 1.0])

def model(x, iv):
    p = iv("xp", np.maximum(x[:, 0], 0.0))
    q = iv("xn", np.maximum(-x[:, 0], 0.0))
    h = p[:, None] * E1 + q[:, None] * E2
    for t in range(1, 4):
        pre = x[:, t:t+1] * SGN + h @ M
        h = iv("u%d" % (t+1), np.maximum(pre, 0.0))
    m = h @ D
    return np.column_stack([np.zeros(len(h)), m])

ALIGN = {"xp": ["h1"], "xn": ["h1"], "u2": ["h2"], "u3": ["h3"], "u4": ["h4"]}
