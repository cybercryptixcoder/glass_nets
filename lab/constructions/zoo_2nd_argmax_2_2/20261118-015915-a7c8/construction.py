DESCRIPTION = ("The network decides which of the two inputs is the smaller (the position of the second-largest). "
               "h1 encodes x1 by its positive and negative parts, relu(x1) and relu(-x1). "
               "h2 forms the two positive differences, relu(x1-x2) and relu(x2-x1). "
               "The output is un - up for class 0 and up - un for class 1, so class 0 wins exactly when x2 > x1 (x1 is the smaller).")

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    vp = iv("h1_pos", np.maximum(x1, 0.0))
    vn = iv("h1_neg", np.maximum(-x1, 0.0))
    x1_rec = vp - vn
    up = iv("h2_up", np.maximum(x1_rec - x2, 0.0))
    un = iv("h2_un", np.maximum(x2 - x1_rec, 0.0))
    out0 = un - up
    out1 = up - un
    return np.stack([out0, out1], axis=1)

ALIGN = {"h1_pos": ["h1"], "h1_neg": ["h1"], "h2_up": ["h2"], "h2_un": ["h2"]}
