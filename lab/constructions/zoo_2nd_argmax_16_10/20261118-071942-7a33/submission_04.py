
import numpy as np
DESCRIPTION = """The network reports the position of the second largest value of the 10 numbers (95.5% agreement,
= its own accuracy). Hypothesis tested here: each hidden state carries (a) a position code for the running
second-largest element of the prefix, (b) the prefix maximum, (c) the prefix second maximum.
model() answers argmax of the final position code."""

def model(x, iv):
    B, T = x.shape
    p1 = iv("pos1", np.maximum(x[:,0], 0.0))
    n1 = iv("neg1", np.maximum(-x[:,0], 0.0))
    P = x[:,0].copy()          # prefix max
    V = np.full(B, -1e9)       # running max of min(x_t, prefix max before t)
    m2 = V.copy()
    code = np.eye(10)[np.zeros(B, dtype=int)]
    code = iv("c1", code); P = iv("P1", P); m2 = iv("m2_1", m2)
    for t in range(1, 10):
        xt = x[:,t]
        V = np.maximum(V, np.minimum(xt, P))
        upd = np.minimum(xt, P) > m2
        m2 = np.where(upd, np.minimum(xt, P), m2)
        code = iv("c%d"%(t+1), np.eye(10)[t]*(upd[:,None]) + code*(~upd)[:,None])
        P = iv("P%d"%(t+1), np.maximum(P, xt))
        m2 = iv("m2_%d"%(t+1), m2)
    return code*1.0

ALIGN = {"pos1":["h1"], "neg1":["h1"]}
for _t in range(2,11):
    ALIGN["c%d"%_t] = ["h%d"%_t]; ALIGN["P%d"%_t] = ["h%d"%_t]; ALIGN["m2_%d"%_t] = ["h%d"%_t]
