
import numpy as np

DESCRIPTION = """
Partial construction. Stage 1 (embedding): the residual stream at position p is
E[token_p] + P[p], where E is a 4x3 token-embedding table and P a 4x3 position table.
Each position gets its own variable of 3 coordinates (the 3 embedding dimensions of
that position), so the embed site is the identity map of the four position vectors.
Later stages are not yet explained; the output is a placeholder.
"""

E = np.array([[-1.5940, -0.9618, 0.2582],
              [ 1.0295,  0.8282, 0.0764],
              [ 0.3151,  0.9424, 0.4281],
              [ 1.3033,  0.1706, 1.3696]])
P = np.array([[ 0.8674, -0.7002, -0.6549],
              [-0.0662, -0.1318,  2.9920],
              [-4.1745,  1.2921,  1.2031],
              [ 1.7989,  1.4152, -0.1054]])

def model(x, iv):
    pos = []
    for p in range(4):
        v = iv("e%d" % p, E[x[:, p], :] + P[p])
        pos.append(v)
    s = pos[3].sum(axis=1)
    return np.stack([s, s, s, s], axis=1)

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"]}
