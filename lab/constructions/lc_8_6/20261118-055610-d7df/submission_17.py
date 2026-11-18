import numpy as np
DESCRIPTION = ("Embedding holds the one-hot token identity at each of the 6 positions; every later quantity is recomputed "
               "from those identities. Read the sequence as a map f(p)=x_p. Layer 0's attention makes each position's "
               "residual carry two structural features: whether that position is a fixed point of the map, and how "
               "many preimages its value has. Layer 1 forms an attention-weighted census of those per-position "
               "features and the readout is linear in that census: a 2-cycle (p->q->p) gives class 1; a fixed point "
               "whose value has another preimage, or two or more fixed points, gives class 0; a lone fixed point or "
               "no fixed point at all gives class 2. Class 3 is a near-tie with class 2 that appears when the last "
               "token is 5 with a single preimage and there is neither a fixed point nor a 2-cycle.")

def model(x, iv):
    B = len(x)
    tok = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        tok.append(iv("tok%d" % p, oh))
    ids = []
    for p in range(6):
        v = np.zeros(B)
        for k in range(6):
            v = v + k * tok[p][:, k]
        ids.append(v)
    T = np.stack(ids, 1)
    m = (np.abs(T - np.arange(6)[None, :]) < 0.5).astype(float)
    fp = iv("fp", m.sum(1))
    cnt = np.stack([(np.abs(T - j) < 0.5).sum(1) for j in range(6)], 1).astype(float)
    pre = iv("pre", cnt)
    t2 = np.zeros(B)
    for p in range(6):
        for q in range(p + 1, 6):
            t2 = t2 + ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5)).astype(float)
    t2 = iv("t2", t2)
    nP = iv("nP", (m * cnt).max(1))
    has1 = (t2 > 0).astype(float)
    has0 = (fp > 0).astype(float)
    lone = (fp == 1) * (nP < 1.5) * (1 - has1)
    s0 = (1 - has1) * (4.0 * has0 * (1 - lone) + 2.0 * fp)
    s1 = 6.0 * has1 + 2.0 * t2
    s2 = (1 - has1) * (3.0 * (1 - has0) + 5.0 * lone)
    last5 = (np.abs(T[:, 5] - 5.0) < 0.5).astype(float)
    one5 = (np.abs(cnt[:, 5] - 1.0) < 0.5).astype(float)
    c3 = (1 - has1) * (1 - has0) * last5 * one5
    s3 = s2 + 0.5 * c3
    score = iv("score", np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1))
    return score

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"],
         "score": ["resid1.mid"]}
