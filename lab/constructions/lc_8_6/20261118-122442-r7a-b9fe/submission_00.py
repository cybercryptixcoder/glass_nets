import numpy as np
DESCRIPTION = ("Embedding holds the one-hot token identity at each of the 6 positions and every later quantity is "
               "recomputed from those identities. Read the sequence as a map f(p)=x_p. Layer-0 attention at query p "
               "selects the key positions j whose value x_j equals p, i.e. the preimages of p, so each position's "
               "residual carries (i) whether that position is a fixed point and (ii) how many preimages its value has. "
               "Layer 1 forms a position-weighted census of those per-position features and the readout is a linear "
               "map of that census: a 2-cycle gives class 1; a fixed point whose value has another preimage (or two or "
               "more fixed points) gives class 0; a lone fixed point or none gives class 2.")

def model(x, iv):
    tok = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        tok.append(iv("tok%d" % p, oh))
    ids = []
    for p in range(6):
        v = np.zeros(len(x))
        for k in range(6):
            v = v + k * tok[p][:, k]
        ids.append(v)
    T = np.stack(ids, 1)                                  # [B,6] token ids
    Tpos = np.arange(6)[None, :]
    m = (np.abs(T - Tpos) < 0.5).astype(float)            # fixed-point mask
    fp = m.sum(1)
    cnt = np.stack([(np.abs(T - j) < 0.5).sum(1) for j in range(6)], 1).astype(float)   # preimages of value j
    t2 = np.zeros(len(x))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 = t2 + ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5)).astype(float)
    fp = iv("fp", fp)
    t2 = iv("t2", t2)
    pre = iv("pre", cnt)
    nP = iv("nP", (m * cnt).max(1))                       # largest preimage count among fixed points
    has1 = (t2 > 0).astype(float)
    has0 = (fp > 0).astype(float)
    lone = (fp == 1) * (nP < 1.5) * (1 - has1)
    s0 = (1 - has1) * (4.0 * has0 * (1 - lone) + 2.0 * fp)
    s1 = 6.0 * has1 + 2.0 * t2
    s2 = (1 - has1) * (3.0 * (1 - has0) + 5.0 * lone)
    s3 = -1.0 + 0.0 * fp
    return np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1)

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
