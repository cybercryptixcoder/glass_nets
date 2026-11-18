
DESCRIPTION = (
 "The hidden state stays positive (ReLU) and becomes almost rank one: h_t ~ s_t * u, with u a fixed positive "
 "direction. The share of h_t's variance on that single axis grows from 74% at step 1 to 99% at step 10, and u is "
 "the mean direction (cosine 0.99997 with mean(h10)). So every site is one scalar, the state magnitude s_t. "
 "Step 1 is exact: h1 = relu(x1*W_x) = pos1*relu(W_x) + neg1*relu(-W_x) with pos1=max(x1,0), neg1=max(-x1,0), and "
 "s1 = 3.99*pos1 + 21.76*neg1. After that the scalar dynamics is a rectified leaky accumulator: "
 "s_{t+1} = relu(a*s_t + b + c*max(x,0) + d*max(-x,0)), with the gain a near 1.7 (the state feeds itself "
 "positively through W_h) and negative inputs adding a lot (W_x is strongly negative on units 6,10,13,14) while "
 "positive inputs add little. The readout of the positive state gives the fixed logit pattern K = u.W_o, in which "
 "class 5 is always largest, so the network always answers class 5."
)

def model(x, iv):
    pos1 = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg1 = iv("neg1", np.maximum(-x[:, 0], 0.0))
    s = iv("s1", 3.9851 * pos1 + 21.7581 * neg1)
    COEF = [
        (0.78, 3.99, 2.10, 13.40),
        (1.96, -2.35, -2.48, 10.30),
        (1.97, -3.64, -7.32, 10.96),
        (1.81, 5.86, -10.22, 11.89),
        (1.66, 18.75, -11.38, 12.49),
        (1.55, 45.88, -13.08, 11.81),
        (1.60, 62.78, -12.51, 11.71),
        (1.66, 67.09, -13.60, 12.95),
        (1.68, 57.56, -10.72, 9.35),
    ]
    for t, (a, b, c, d) in enumerate(COEF, start=1):
        u = x[:, t]
        s = np.maximum(a * s + b + c * np.maximum(u, 0.0) + d * np.maximum(-u, 0.0), 0.0)
        s = iv("s%d" % (t + 1), s)
    K = np.array([-2.42, -2.48, -0.03, -2.22, 0.60, 3.88, 1.64, 0.03, -1.33, -0.17])
    return s[:, None] * K[None, :]

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "s1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"],
         "s5": ["h5"], "s6": ["h6"], "s7": ["h7"], "s8": ["h8"], "s9": ["h9"], "s10": ["h10"]}
