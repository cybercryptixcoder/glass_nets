import numpy as np
DESCRIPTION = ("Embedding holds the one-hot token identity at each of the 6 positions; every later quantity is recomputed "
               "from those identities. Reading the sequence as a map f(p)=x_p, the network's layer-0 attention at query p "
               "selects the key positions j whose value x_j equals p (the preimages of p), so each position's residual "
               "carries whether that position is a fixed point. Layer 1 mixes those per-position features at the last "
               "position, so the decision is a weighted census of fixed points and 2-cycles: a 2-cycle gives class 1, a "
               "fixed point gives class 0 (weighted by how much layer 1 attends to that position), and neither gives "
               "class 2. This readout is the resulting rule.")

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
    t2 = np.zeros(len(x))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 = t2 + ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5)).astype(float)
    fp = iv("fp", fp)
    t2 = iv("t2", t2)
    w = np.array([0.42, 0.11, 0.22, 0.13, 0.00, 0.12])    # layer-1 attention weight by position
    census = m @ w                                        # attention-weighted fixed-point census
    census = iv("census", census)
    has1 = (t2 > 0).astype(float)
    has0 = (fp > 0).astype(float)
    low = (fp == 1) * (census < 0.115) * (1 - has1)       # the fixed point sits where layer 1 does not look
    s0 = 4.0 * has0 * (1 - has1) * (1 - low) + 2.0 * fp
    s1 = 6.0 * has1 + 2.0 * t2
    s2 = 3.0 * (1 - has0) * (1 - has1) + 5.0 * low
    s3 = -1.0 + 0.0 * fp
    return np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1)

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
