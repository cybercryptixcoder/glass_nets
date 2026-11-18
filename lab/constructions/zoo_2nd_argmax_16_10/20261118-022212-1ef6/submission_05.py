
DESCRIPTION = (
 "The state stays non-negative (ReLU) and becomes almost rank one: h_t ~ s_t*u with u a fixed positive "
 "direction (variance share on that axis 0.74 at step 1 rising to 0.99 at step 10); u is the mean direction "
 "(cosine 0.99997 with mean(h10)). Every site is therefore one number, the state magnitude s_t, and the readout "
 "of that positive state is the fixed pattern K = u.W_o, whose class-5 entry is largest: the network always "
 "answers class 5 (the network is wrong on ~90% of its inputs; this construction reproduces its answers, not the "
 "task). Step 1 is exact: h1 = relu(x1*W_x) = pos1*relu(W_x) + neg1*relu(-W_x), i.e. s1 = 3.99*pos1 + 21.76*neg1 "
 "with pos=max(x,0), neg=max(-x,0). Afterwards the magnitude follows a rectified leaky accumulator "
 "s_{t+1} = relu(a*s_t + b + c*pos_{t-1} + d*neg_{t-1} + e*pos_{t-2} + f*neg_{t-2}): gain a ~ 1.6-2.0 (the "
 "state feeds itself positively through W_h), negative inputs add strongly (W_x is negative on units 6,10,13,14), "
 "positive inputs add little; the constant term grows with t because the mean state is amplified step after step."
)

def model(x, iv):
    pos = np.maximum(x, 0.0)
    neg = np.maximum(-x, 0.0)
    s = iv("s1", 3.9851 * pos[:, 0] + 21.7581 * neg[:, 0])
    COEF = [
        (0.93,  6.55,  6.43, -1.14,  0.00,  0.00),
        (0.83,  3.59,  8.55,  3.70, -2.67, 34.64),
        (2.01, -1.48,  6.46, -7.85, -14.37, 11.45),
        (1.82,  3.49,  7.23, -8.40, -2.75,  9.91),
        (1.67, 14.84,  6.98, -8.24,  2.68,  7.65),
        (1.55, 41.41,  7.27, -6.80,  1.29,  7.26),
        (1.60, 61.57,  6.86, -6.27, -0.91,  2.51),
        (1.66, 64.63,  6.69, -6.38,  1.28,  2.74),
        (1.68, 54.02,  6.11, -6.01,  2.02,  3.84),
    ]
    for t, (a, b, c, d, e, f) in enumerate(COEF, start=1):
        j1, j2 = t - 1, t - 2
        p1 = pos[:, j1]; n1 = neg[:, j1]
        p2 = pos[:, j2] if j2 >= 0 else 0.0
        n2 = neg[:, j2] if j2 >= 0 else 0.0
        s = np.maximum(a * s + b + c * p1 + d * n1 + e * p2 + f * n2, 0.0)
        s = iv("s%d" % (t + 1), s)
    K = np.array([-2.42, -2.48, -0.03, -2.22, 0.60, 3.88, 1.64, 0.03, -1.33, -0.17])
    return s[:, None] * K[None, :]

ALIGN = {"s1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"],
         "s6": ["h6"], "s7": ["h7"], "s8": ["h8"], "s9": ["h9"], "s10": ["h10"]}
