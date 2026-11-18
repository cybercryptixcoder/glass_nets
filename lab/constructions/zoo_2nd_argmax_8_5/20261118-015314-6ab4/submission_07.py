import numpy as np

DESCRIPTION = """
Stage 'step 1' is explained: the first input weight row has a fixed sign pattern (units 1,3,5
negative, units 0,2,4,6,7 positive), and with no biases
    h1 = relu(x1)*W_x+ + relu(-x1)*W_x-
so the first state holds the positive and negative parts of input 1 in two disjoint coordinate
groups: a sign-magnitude code of the value it has just read.
Later stages are modelled as running order statistics of the prefix: each step keeps the running
largest and running second largest of the inputs read so far, again in sign-magnitude form
(relu part in the 'positive' units, relu of the negative part in the negative units).
The readout scores each position c with
   out_c = 12*relu(x_c - s2_c) - 24*relu(x_c - m_c) - 6*relu(s2_c - x_c)
where m_c, s2_c are the largest and second largest of the four inputs outside position c: a
position wins when it is above everything except one other input, i.e. the second largest.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def stats(vals):
    """largest and second largest along axis 0, values shaped [B, n]."""
    m = np.max(vals, axis=1)
    if vals.shape[1] == 1:
        s = np.full_like(m, -1e9)
    else:
        s = np.sort(vals, axis=1)[:, -2]
    return m, s


def order_out(x):
    out = np.zeros((x.shape[0], 5))
    for c in range(5):
        others = np.delete(x, c, axis=1)
        mo, so = stats(others)
        xc = x[:, c]
        out[:, c] = (12.0 * np.maximum(xc - so, 0.0)
                     - 24.0 * np.maximum(xc - mo, 0.0)
                     - 6.0 * np.maximum(so - xc, 0.0))
    return out


def model(x, iv):
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]
    x1 = pos - neg

    vals = x1[:, None]
    mm = iv("mm", np.maximum(np.max(vals, axis=1), 0.0))
    nm = iv("nm", np.maximum(-np.max(vals, axis=1), 0.0))
    ssum = np.full(x.shape[0], -1e9)
    ss = iv("ss", np.maximum(ssum, 0.0))
    ns = iv("ns", np.maximum(-ssum, 0.0))
    mm, nm, ss, ns = iv("mm", mm), iv("nm", nm), iv("ss", ss), iv("ns", ns)
    m = mm - nm
    s = ss - ns

    for t in range(1, 5):
        xt = x[:, t]
        newm = np.maximum(m, xt)
        news = np.maximum(np.minimum(m, xt), s)
        m, s = newm, news
        if t == 1:
            mm, nm = iv("mm2", np.maximum(m, 0.0)), iv("nm2", np.maximum(-m, 0.0))
            ss, ns = iv("ss2", np.maximum(s, 0.0)), iv("ns2", np.maximum(-s, 0.0))
        elif t == 2:
            mm, nm = iv("mm3", np.maximum(m, 0.0)), iv("nm3", np.maximum(-m, 0.0))
            ss, ns = iv("ss3", np.maximum(s, 0.0)), iv("ns3", np.maximum(-s, 0.0))
        elif t == 3:
            mm, nm = iv("mm4", np.maximum(m, 0.0)), iv("nm4", np.maximum(-m, 0.0))
            ss, ns = iv("ss4", np.maximum(s, 0.0)), iv("ns4", np.maximum(-s, 0.0))
        else:
            mm, nm = iv("mm5", np.maximum(m, 0.0)), iv("nm5", np.maximum(-m, 0.0))
            ss, ns = iv("ss5", np.maximum(s, 0.0)), iv("ns5", np.maximum(-s, 0.0))
        m = mm - nm
        s = ss - ns

    seq = np.concatenate([x1[:, None], x[:, 1:]], axis=1)
    return order_out(seq)


ALIGN = {"pos1": ["h1"], "neg1": ["h1"],
         "mm2": ["h2"], "nm2": ["h2"], "ss2": ["h2"], "ns2": ["h2"],
         "mm3": ["h3"], "nm3": ["h3"], "ss3": ["h3"], "ns3": ["h3"],
         "mm4": ["h4"], "nm4": ["h4"], "ss4": ["h4"], "ns4": ["h4"],
         "mm5": ["h5"], "nm5": ["h5"], "ss5": ["h5"], "ns5": ["h5"]}
