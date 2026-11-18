
import numpy as np
DESCRIPTION = "Embedding = one-hot token identity at each of the 6 positions. First structural finding: the class-0 logit is driven by positions where x_p == p (fixed points of the sequence read as a map p->x_p); class-0 probability rises with the number of fixed points (fp=0: 2%, fp=1: 43%, fp=2: 73%, fp=3: 88%). With no fixed point the network mostly outputs 1 or 2. Readout here is the crude rule 0 if any fixed point else 2."
def model(x, iv):
    toks = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        toks.append(iv("tok%d" % p, oh))
    fp = iv("fp", (x == np.arange(6)[None, :]).sum(1).astype(float))
    s0 = 5.0 * fp - 1.0
    s1 = 0.0 * fp
    s2 = 0.5 - 0.0 * fp
    s3 = -2.0 - 0.0 * fp
    return np.stack([s0, s1, s2, s3, s3 - 1.0, s3 - 2.0], 1)
ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
