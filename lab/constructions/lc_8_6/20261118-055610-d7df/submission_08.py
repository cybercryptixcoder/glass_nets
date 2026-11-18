
import numpy as np
DESCRIPTION = "Embedding holds the one-hot token identity at each of the 6 positions. Structural finding: reading the sequence as a map f(p)=x_p, the network separates three cases. If f has a 2-cycle (p!=q with x_p=q, x_q=p) the network outputs class 1 (98%); if f has a fixed point (x_p=p) and no 2-cycle it outputs class 0 (81%, rising with the number of fixed points); if f has neither it outputs class 2 (91%). Readout below is this rule, roughly graded."
def model(x, iv):
    toks = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        toks.append(iv("tok%d" % p, oh))
    T = x.astype(int)
    fp = (T == np.arange(6)[None, :]).sum(1)
    t2 = np.zeros(len(x))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 += ((T[:, p] == q) & (T[:, q] == p))
    fp = iv("fp", fp.astype(float)); t2 = iv("t2", t2)
    has1 = (t2 > 0).astype(float); has0 = (fp > 0).astype(float)
    s0 = 4 * has0 * (1 - has1) + 2 * fp
    s1 = 6 * has1 + 2 * t2
    s2 = 3 * (1 - has0) * (1 - has1)
    s3 = -1.0 + 0 * fp
    return np.stack([s0, s1, s2, s3, s3 - 1, s3 - 2], 1)
ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
