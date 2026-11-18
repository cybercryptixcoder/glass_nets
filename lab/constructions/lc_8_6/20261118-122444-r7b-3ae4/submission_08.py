
import numpy as np
DESCRIPTION = (
 "Token identity at each position lives in the embedding (six one-hots). Everything later is recomputed "
 "from those identities. Per position p the residual carries a census: whether p is a fixed point "
 "(x_p==p), how many preimages the value x_p has, and whether p sits on a 2-cycle. The layer-1 read-out at "
 "the last position is a linear map of that census: a 2-cycle -> class 1; a fixed point whose value has "
 "another preimage, or two or more fixed points -> class 0; a lone fixed point or none -> class 2; class 3 "
 "is the rare no-fixed-point/no-2-cycle cell.")

def model(x, iv):
    tok = []
    for p in range(6):
        tok.append(iv("tok%d" % p, (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)))
    T = np.stack([sum(k * tok[p][:, k] for k in range(6)) for p in range(6)], 1)
    pos = np.arange(6)[None, :]
    m = (np.abs(T - pos) < 0.5).astype(float)
    cnt = np.stack([(np.abs(T - j) < 0.5).sum(1) for j in range(6)], 1).astype(float)
    t2 = np.zeros(len(x))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 = t2 + ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5)).astype(float)
    fp = iv("fp", m.sum(1))
    t2 = iv("t2", t2)
    pre = iv("pre", cnt)
    nP = iv("nP", (m * cnt).max(1))
    has1 = (t2 > 0).astype(float); has0 = (fp > 0).astype(float)
    lone = (fp == 1) * (nP < 1.5) * (1 - has1)
    s0 = (1 - has1) * (4.0 * has0 * (1 - lone) + 2.0 * fp)
    s1 = 6.0 * has1 + 2.0 * t2
    s2 = (1 - has1) * (3.0 * (1 - has0) + 5.0 * lone)
    s3 = -1.0 + 0.0 * fp
    logits = iv("logits", np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1))
    return logits

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"],
         "fp": ["resid0.mid"], "pre": ["resid0.mid"], "nP": ["resid0.mid"], "t2": ["resid0.mid"],
         "logits": ["resid1.mid"]}
