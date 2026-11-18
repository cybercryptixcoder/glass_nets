import numpy as np
DESCRIPTION = """Semantic candidate: the network answers 'which position holds the second largest value'
(95.3% agreement = its own accuracy). The state is modelled by running summaries of the prefix:
m1 = largest value so far, m2 = second largest value so far, and the answer candidate's position held as a
phase pair (cos, sin) of 2*pi*p/10 -- the readout of the network is dominated by class-frequency 1, so a
phase code is the natural carrier. The output scores are the projection of that phase onto the ten
position phases, so the largest score sits at the answer position."""
TAU = 6.283185307179586

def model(x, iv):
    B = x.shape[0]
    m1 = np.full(B, -1e9); m2 = np.full(B, -1e9)
    p2 = np.zeros(B)
    c = np.ones(B); s = np.zeros(B)
    for t in range(10):
        xt = x[:, t]
        newmax = xt > m1
        # demote old max to second, else it may enter second place
        m2n = np.where(newmax, m1, np.maximum(m2, xt))
        p2n = np.where(newmax, p2, np.where(xt > m2, t, p2))
        m1n = np.where(newmax, xt, m1)
        m1 = iv("m1_%d" % (t+1), m1n)
        m2 = iv("m2_%d" % (t+1), m2n)
        p2 = p2n
        ang = TAU * p2 / 10.0
        c = iv("cos_%d" % (t+1), np.cos(ang))
        s = iv("sin_%d" % (t+1), np.sin(ang))
    th = TAU * np.arange(10) / 10.0
    return c[:, None] * np.cos(th)[None, :] + s[:, None] * np.sin(th)[None, :]

ALIGN = {}
for _t in range(1, 11):
    for _v in ("m1_%d", "m2_%d", "cos_%d", "sin_%d"):
        ALIGN[_v % _t] = ["h%d" % _t]
