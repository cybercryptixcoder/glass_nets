import numpy as np
DESCRIPTION = ("The hidden state is two signed channels, each stored as a positive and a negative ReLU part: "
               "unit1=relu(v1), unit2=relu(v2), unit3=relu(-v2), unit4=relu(-v1). Each step adds the new input "
               "x_t times a per-unit gain and applies the linear recurrent map, then ReLU: "
               "h_t = relu(x_t*g + h_{t-1}@M). Step 1 therefore holds the pair (3.97*x1, 4.70*x1) with sign. "
               "The readout sees only the MAGNITUDES of the two channels, m1=h1+h4, m2=h2+h3: "
               "out = w0*m1 + w1*m2, i.e. three classes are three fixed weightings of the two magnitudes and "
               "the decision is the largest. (v-basis and readout measured; M written explicitly.)")
def model(x, iv):
    g = np.array([3.9653, 4.6995, -4.7163, -3.9652])
    M = np.array([[1.3387, 1.9277, -2.3877, -1.8931],
                  [-2.1996, -2.4339, 2.8057, 2.6429],
                  [2.6352, 2.7868, -2.4339, -2.1912],
                  [-1.8948, -2.3826, 1.9331, 1.3409]])
    Wo = np.array([[-1.66465, -18.73600, 11.45370],
                   [1.60840, 15.97925, -15.71880]])
    h = np.maximum(x[:, 0:1] * g, 0.0)
    h = iv("u1", h[:, 0]), h
    h1 = np.stack([iv("h1_u1", h[1][:, 0]), iv("h1_u2", h[1][:, 1]),
                   iv("h1_u3", h[1][:, 2]), iv("h1_u4", h[1][:, 3])], axis=1)
    pre2 = x[:, 1:2] * g + h1 @ M
    h2 = np.stack([iv("h2_u1", np.maximum(pre2[:, 0], 0.0)),
                   iv("h2_u2", np.maximum(pre2[:, 1], 0.0)),
                   iv("h2_u3", np.maximum(pre2[:, 2], 0.0)),
                   iv("h2_u4", np.maximum(pre2[:, 3], 0.0))], axis=1)
    pre3 = x[:, 2:3] * g + h2 @ M
    h3 = np.stack([iv("h3_u1", np.maximum(pre3[:, 0], 0.0)),
                   iv("h3_u2", np.maximum(pre3[:, 1], 0.0)),
                   iv("h3_u3", np.maximum(pre3[:, 2], 0.0)),
                   iv("h3_u4", np.maximum(pre3[:, 3], 0.0))], axis=1)
    m1 = iv("m1", h3[:, 0] + h3[:, 3])
    m2 = iv("m2", h3[:, 1] + h3[:, 2])
    return np.stack([m1, m2], axis=1) @ Wo.T
ALIGN = {"h1_u1": ["h1"], "h1_u2": ["h1"], "h1_u3": ["h1"], "h1_u4": ["h1"],
         "h2_u1": ["h2"], "h2_u2": ["h2"], "h2_u3": ["h2"], "h2_u4": ["h2"],
         "h3_u1": ["h3"], "h3_u2": ["h3"], "h3_u3": ["h3"], "h3_u4": ["h3"]}
