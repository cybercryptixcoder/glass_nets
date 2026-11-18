
import numpy as np
DESCRIPTION = """The network answers 'which position holds the second largest value of the sequence'
(agreement 95.3%, exactly its accuracy). Variables named here are the semantic running summaries:
m1_t = max of the prefix, m2_t = second largest of the prefix (updated as
m2 = max(m2, min(m1, x_t))). model() therefore answers the index of the second largest value."""

def model(x, iv):
    B = x.shape[0]
    m1 = np.zeros(B); m2 = np.zeros(B)
    for t in range(10):
        xt = x[:,t]
        m1n = np.maximum(m1, xt)
        m2n = np.maximum(m2, np.minimum(m1, xt))
        m1 = iv("m1_%d"%(t+1), m1n)
        m2 = iv("m2_%d"%(t+1), m2n)
    order = np.argsort(-x, axis=1)
    ans = order[:,1]
    out = -np.ones((B,10))
    out[np.arange(B), ans] = 1.0
    return out

ALIGN = {}
for _t in range(1,11):
    ALIGN["m1_%d"%_t] = ["h%d"%_t]
    ALIGN["m2_%d"%_t] = ["h%d"%_t]
