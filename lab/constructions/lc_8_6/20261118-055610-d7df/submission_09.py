import numpy as np
DESCRIPTION = ("Embedding holds the one-hot token identity at each of the 6 positions; everything later is computed "
               "from those identities. Reading the sequence as a map f(p)=x_p: if f has a 2-cycle (p!=q with x_p=q, x_q=p) "
               "the network outputs class 1; if f has a fixed point (x_p=p) and no 2-cycle it outputs class 0; if it has "
               "neither it outputs class 2. The construction below re-derives every later quantity from the token variables "
               "it is given, so intervening on the tokens changes its decision the way intervening on the embedding does.")

def model(x, iv):
    tok = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        tok.append(iv("tok%d" % p, oh))
    # token ids recovered from the (possibly intervened) token variables
    T = np.zeros(len(x))
    for p in range(6):
        for k in range(6):
            T = T + 0.0 * tok[p][:, k] + k * tok[p][:, k] * 0.0
    # proper: per-position ids
    ids = []
    for p in range(6):
        v = np.zeros(len(x))
        for k in range(6):
            v = v + k * tok[p][:, k]
        ids.append(v)
    T = np.stack(ids, 1)                      # [B,6]
    Tpos = np.arange(6)[None, :]
    m = (np.abs(T - Tpos) < 0.5).astype(float)          # fixed-point mask
    fp = m.sum(1)
    t2 = np.zeros(len(x))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 = t2 + ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5)).astype(float)
    fp = iv("fp", fp)
    t2 = iv("t2", t2)
    has1 = (t2 > 0).astype(float)
    has0 = (fp > 0).astype(float)
    s0 = 4.0 * has0 * (1 - has1) + 2.0 * fp - 4.0 * has1
    s1 = 6.0 * has1 + 2.0 * t2
    s2 = 3.0 * (1 - has0) * (1 - has1)
    s3 = -1.0 + 0.0 * fp
    return np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1)

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
