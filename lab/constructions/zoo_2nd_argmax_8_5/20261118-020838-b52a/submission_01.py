import numpy as np

DESCRIPTION = (
    "Running top-2 tracker. After reading each input the state holds the two largest "
    "values seen so far, each split into a non-negative part and a negative part "
    "(m1p/m1n = max, m2p/m2n = second max), together with one-hot position codes "
    "p1 (position of the running maximum) and p2 (position of the running second "
    "maximum). A new input above m1 demotes the old maximum into the second slot and "
    "takes the lead; an input between m2 and m1 becomes the new second; anything "
    "smaller changes nothing. Each step is exactly this comparison-and-swap. The "
    "output is a readout of the final second-maximum position, so the construction "
    "returns p2 (scaled)."
)


def model(x, iv):
    x = np.asarray(x, dtype=float)
    B, T = x.shape
    SENT = -10.0          # 'no second maximum yet' (below any input)

    def split(v):
        return np.maximum(v, 0.0), np.maximum(-v, 0.0)

    # ---- step 1: only one value seen ----
    m1 = x[:, 0].copy()
    m2 = np.full(B, SENT)
    p1 = np.zeros((B, T))
    p1[:, 0] = 1.0
    p2 = np.zeros((B, T))
    a, b = split(m1)
    c, d = split(m2)
    m1p = iv("m1p", a); m1n = iv("m1n", b)
    m2p = iv("m2p", c); m2n = iv("m2n", d)
    p1 = iv("p1", p1); p2 = iv("p2", p2)
    m1 = m1p - m1n
    m2 = m2p - m2n

    # ---- steps 2..T: compare, insert, demote ----
    for t in range(1, T):
        xt = x[:, t]
        oh = np.zeros((B, T))
        oh[:, t] = 1.0
        g1 = (xt > m1).astype(float)          # new global maximum
        g2 = (xt > m2).astype(float)          # beats the second maximum
        n1 = g1 * xt + (1.0 - g1) * m1
        n2 = g1 * m1 + (1.0 - g1) * (g2 * xt + (1.0 - g2) * m2)
        q1 = g1[:, None] * oh + (1.0 - g1)[:, None] * p1
        q2 = (g1[:, None] * p1
              + (1.0 - g1)[:, None] * (g2[:, None] * oh + (1.0 - g2)[:, None] * p2))
        a, b = split(n1)
        c, d = split(n2)
        m1p = iv("m1p", a); m1n = iv("m1n", b)
        m2p = iv("m2p", c); m2n = iv("m2n", d)
        p1 = iv("p1", q1); p2 = iv("p2", q2)
        m1 = m1p - m1n
        m2 = m2p - m2n

    return 10.0 * p2


_SITES = ["h1", "h2", "h3", "h4", "h5"]
ALIGN = {name: list(_SITES) for name in ("m1p", "m1n", "m2p", "m2n", "p1", "p2")}
