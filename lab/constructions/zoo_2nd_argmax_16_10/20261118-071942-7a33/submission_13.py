import numpy as np
DESCRIPTION = """Step 1 is explained exactly: h1 = relu(x1*W_x) = x1^+ * max(W_x,0) + x1^- * max(-W_x,0), a
2-dimensional object, so I name the two numbers pos1 = x1^+ and neg1 = x1^- there. The network's task is
'which position holds the second largest of the ten values' (it agrees with that rule on 95.3% of inputs,
its own accuracy); its residual errors are near-ties between the 2nd and 3rd largest (median gap 0.032 on
errors, 0.27 when right) and it then keeps the earlier position. The readout is therefore a running top-2
scan that needs a small margin before a stored value is displaced. At the later sites I name the running
top-2 values as a probe of how much of the state is 'top-2 content'."""

MARGIN = 0.01

def model(x, iv):
    B = x.shape[0]
    pos1 = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg1 = iv("neg1", np.maximum(-x[:, 0], 0.0))
    del pos1, neg1
    v1 = np.full(B, -1000000000.0)
    v2 = np.full(B, -1000000000.0)
    p1 = np.zeros(B, dtype=int)
    p2 = np.zeros(B, dtype=int)
    t = 0
    while t < 10:
        xt = x[:, t]
        newtop = xt > v1 + MARGIN
        newsec = (xt > v2 + MARGIN) & (v1 >= xt - MARGIN)
        p2 = np.where(newtop, p1, np.where(newsec, t, p2))
        v2 = np.where(newtop, v1, np.where(newsec, xt, v2))
        p1 = np.where(newtop, t, p1)
        v1 = np.where(newtop, xt, v1)
        if t >= 1:
            v2 = iv("v2_%d" % (t + 1), v2)
            v1 = iv("v1_%d" % (t + 1), v1)
        t += 1
    out = -np.ones((B, 10))
    out[np.arange(B), p2] = 1.0
    return out

ALIGN = {"pos1": ["h1"], "neg1": ["h1"],
         "v1_2": ["h2"], "v2_2": ["h2"], "v1_3": ["h3"], "v2_3": ["h3"],
         "v1_4": ["h4"], "v2_4": ["h4"], "v1_5": ["h5"], "v2_5": ["h5"],
         "v1_6": ["h6"], "v2_6": ["h6"], "v1_7": ["h7"], "v2_7": ["h7"],
         "v1_8": ["h8"], "v2_8": ["h8"], "v1_9": ["h9"], "v2_9": ["h9"],
         "v1_10": ["h10"], "v2_10": ["h10"]}
