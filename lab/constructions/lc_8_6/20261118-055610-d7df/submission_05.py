
import numpy as np
DESCRIPTION = "Work in progress: embedding holds one-hot token identity at each of the 6 positions; the readout is still being identified."
def model(x, iv):
    toks = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        toks.append(iv("tok%d" % p, oh))
    return np.stack(toks, 1).sum(1)
ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"]}
