import numpy as np

DESCRIPTION = (
    "The network tracks the two largest values seen so far. At every step its 16 hidden "
    "units hold (i) a 10-slot rank code c: c_i=1 if position i is the current maximum, "
    "c_i=2 if position i is the current second maximum, 0 otherwise; and (ii) the signed "
    "values m (max) and s (second max), stored as positive/negative ReLU parts because the "
    "hidden state is non-negative. Step 1 can only encode x1, and indeed h1 = relu(x1 W_x) "
    "is exactly relu(x1)*W_x^+ + relu(-x1)*|W_x^-|, a rank-2 object. Each later step compares "
    "the new x_t with m and s: a new record moves the old max to second, otherwise a value "
    "above s becomes the new second. The readout h10 @ W_o selects the code units, so the "
    "class whose code is 2 (the second-largest position) has the largest score."
)

def model(x, iv):
    B = x.shape[0]
    # Step 1: h1 = relu(x1 W_x) is linear in the positive and negative parts of x1.
    p1 = iv("p1", np.maximum(x[:, 0], 0.0))
    n1 = iv("n1", np.maximum(-x[:, 0], 0.0))
    m = p1 - n1
    s = np.full(B, -1e9)
    code = np.zeros((B, 10))
    code[:, 0] = 1.0

    for t in range(2, 11):
        xt = x[:, t - 1]
        g = (xt > m)          # new record
        h = (xt > s)          # enters top two
        nm = np.where(g, xt, m)
        ns = np.where(g, m, np.where(h, xt, s))

        onehot = np.zeros((B, 10))
        onehot[:, t - 1] = 1.0
        # If new record: old max -> second (1->2), old second dropped, new pos -> max (1).
        case_g = 2.0 * (code == 1.0) + onehot
        # Else if above second: old second dropped (2->0), old max stays 1, new pos -> second (2).
        case_h = np.where(code == 2.0, 0.0, code) + 2.0 * onehot
        ncode = np.where(g[:, None], case_g, np.where(h[:, None], case_h, code))

        # Register the state at this site; later steps must use the returned values.
        mp = iv("h%d_mp" % t, np.maximum(nm, 0.0))
        mn = iv("h%d_mn" % t, np.maximum(-nm, 0.0))
        sp = iv("h%d_sp" % t, np.maximum(ns, 0.0))
        sn = iv("h%d_sn" % t, np.maximum(-ns, 0.0))
        c = iv("h%d_code" % t, ncode)

        m = mp - mn
        s = sp - sn
        code = c

    return code

ALIGN = {"p1": ["h1"], "n1": ["h1"]}
for t in range(2, 11):
    ALIGN["h%d_mp" % t] = ["h%d" % t]
    ALIGN["h%d_mn" % t] = ["h%d" % t]
    ALIGN["h%d_sp" % t] = ["h%d" % t]
    ALIGN["h%d_sn" % t] = ["h%d" % t]
    ALIGN["h%d_code" % t] = ["h%d" % t]
