
import numpy as np
DESCRIPTION = "Temporary construction: embedding holds the token identity at each of the 6 positions (one-hot) and the token counts. Downstream readout still a placeholder."
def model(x, iv):
    toks = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        toks.append(iv("tok%d" % p, oh))
    counts = iv("counts", np.stack(toks, 1).sum(1))
    return counts
ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "tok5": ["embed"],
         "counts": ["attn0.out", "attn0.pattern", "attn1.out", "attn1.pattern"]}
