import numpy as np
DESCRIPTION = ("The hidden state is two signed channels, each stored as a positive and a negative ReLU part: "
               "unit1=relu(v1), unit2=relu(v2), unit3=relu(-v2), unit4=relu(-v1). Each step adds the new input "
               "x_t times a per-unit gain and applies the linear recurrent map, then ReLU: "
               "h_t = relu(x_t*g + h_{t-1}@M). The readout sees only the MAGNITUDES of the two channels, "
               "m1=h1+h4, m2=h2+h3: out = w0*m1 + w1*m2, i.e. the three classes are three fixed weightings of "
               "the two magnitudes and the decision is the largest.")
def model(x, iv):
    g = np.array([3.9653, 4.6995, -4.7163, -3.9652])
    M = np.array([[1.3387, 1.9277, -2.3877, -1.8931],
                  [-2.1996, -2.4339, 2.8057, 2.6429],
                  [2.6352, 2.7868, -2.4339, -2.1912],
                  [-1.8948, -2.3826, 1.9331, 1.3409]])
    Wo = np.array([[-1.66465, -18.73600, 11.45370],
                   [1.60840, 15.97925, -15.71880]])
    p1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    h = np.stack([3.9653 * p1, 4.6995 * p1, 4.7163 * n1, 3.9652 * n1], axis=1)
    h = np.stack([iv("h1_u1", h[:, 0]), iv("h1_u2", h[:, 1]), iv("h1_u3", h[:, 2]), iv("h1_u4", h[:, 3])], axis=1)
    for t in (1, 2):
        pre = x[:, t:t+1] * g + h @ M
        h = np.stack([iv("h%d_u%d" % (t+1, k+1), np.maximum(pre[:, k], 0.0)) for k in range(4)], axis=1)
    m1 = iv("m1", h[:, 0] + h[:, 3])
    m2 = iv("m2", h[:, 1] + h[:, 2])
    return np.stack([m1, m2], axis=1) @ Wo
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"],
         "h1_u1": ["h1"], "h1_u2": ["h1"], "h1_u3": ["h1"], "h1_u4": ["h1"],
         "h2_u1": ["h2"], "h2_u2": ["h2"], "h2_u3": ["h2"], "h2_u4": ["h2"],
         "h3_u1": ["h3"], "h3_u2": ["h3"], "h3_u3": ["h3"], "h3_u4": ["h3"],
         "m1": [], "m2": []}
