import numpy as np
DESCRIPTION = ("The hidden state is two signed channels, each stored as a positive part and a negative part: "
               "unit1=relu(v1), unit4=relu(-v1) (channel 1) and unit2=relu(v2), unit3=relu(-v2) (channel 2); the "
               "network is mirror-symmetric under (unit1<->unit4, unit2<->unit3). Each step each channel gets the "
               "new input (channel 1 gain 3.97, channel 2 gain 4.70, with the opposite sign in the negative part), "
               "plus a linear mixing of the previous state's magnitudes m=(u1+u4, u2+u3) and signed values "
               "s=(u1-u4, u2-u3): the signed part is rotated by ~120 degrees each step, and the magnitude part is a "
               "small rank-one coupling. Then ReLU. So h_t = relu( C[m1,s1,m2,s2] + g*x_t ) with channel 2's rows "
               "the mirror of channel 1's. The readout uses only the two magnitudes: out = w0*m1 + w1*m2, three "
               "fixed weightings of the two channel magnitudes, decision = largest.")
def model(x, iv):
    g = np.array([3.9653, 4.6995, -4.7163, -3.9652])
    C1 = np.array([-0.2780, 1.6167, 0.2178, -2.4174])
    C2 = np.array([-0.2274, 2.1552, 0.1764, -2.6103])
    flip = np.array([1.0, -1.0, 1.0, -1.0])
    CT = np.stack([C1, C2, C2 * flip, C1 * flip])
    W0 = np.array([-1.66465, -18.73600, 11.45370])
    W1 = np.array([1.60840, 15.97925, -15.71880])
    p1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    u = np.stack([g[0] * p1, g[1] * p1, -g[2] * n1, -g[3] * n1], axis=1)
    for t in (1, 2):
        ms = np.stack([u[:, 0] + u[:, 3], u[:, 0] - u[:, 3], u[:, 1] + u[:, 2], u[:, 1] - u[:, 2]], axis=1)
        pre = ms @ CT.T + x[:, t:t + 1] * g
        u = np.stack([iv("h%d_u%d" % (t + 1, k + 1), np.maximum(pre[:, k], 0.0)) for k in range(4)], axis=1)
    m1 = iv("m1", u[:, 0] + u[:, 3])
    m2 = iv("m2", u[:, 1] + u[:, 2])
    return np.stack([m1, m2], axis=1) @ np.stack([W0, W1])
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"],
         "h2_u1": ["h2"], "h2_u2": ["h2"], "h2_u3": ["h2"], "h2_u4": ["h2"],
         "h3_u1": ["h3"], "h3_u2": ["h3"], "h3_u3": ["h3"], "h3_u4": ["h3"]}
