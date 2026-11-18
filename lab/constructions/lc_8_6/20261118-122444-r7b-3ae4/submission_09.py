
import numpy as np
DESCRIPTION = (
 "The embedding holds the six token one-hots and nothing later is read from the input directly. Read x as a map "
 "f(p)=x_p on 0..5. Census recomputed from the identities: p is a fixed point iff x_p==p; a 2-cycle is a pair p<q "
 "with x_p==q and x_q==p; a value j has its preimage count. Scores: a 2-cycle -> class 1; a fixed point whose "
 "value has another preimage, or two or more fixed points -> class 0; a lone fixed point or no fixed point -> "
 "class 2; class 3 is the small unused cell. The read-out at the last position is a linear map of that census.")

def model(x, iv):
    tok = [iv("tok%d" % p, (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)) for p in range(6)]
    T = np.stack([sum(k * tok[p][:, k] for k in range(6)) for p in range(6)], 1)
    m = (np.abs(T - np.arange(6)[None, :]) < 0.5).astype(float)
    cnt = np.stack([(np.abs(T - j) < 0.5).sum(1) for j in range(6)], 1).astype(float)
    t2 = np.zeros(len(x))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 = t2 + ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5)).astype(float)
    fp = m.sum(1)
    has1 = (t2 > 0).astype(float); has0 = (fp > 0).astype(float)
    lone = (fp == 1) * ((m * cnt).max(1) < 1.5) * (1 - has1)
    s0 = (1 - has1) * (4.0 * has0 * (1 - lone) + 2.0 * fp)
    s1 = 6.0 * has1 + 2.0 * t2
    s2 = (1 - has1) * (3.0 * (1 - has0) + 5.0 * lone)
    s3 = -1.0 + 0.0 * fp
    return np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1)

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
