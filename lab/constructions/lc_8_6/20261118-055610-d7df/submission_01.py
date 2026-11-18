
import numpy as np
DESCRIPTION = "Rough first construction: embedding holds the token identity at each of the 6 positions (one-hot), and the tokens' counts. The rest is a placeholder majority-count readout."
def model(x, iv):
    B = x.shape[0]
    toks = []
    for p in range(6):
        oh = (x[:, p][:, None] == np.arange(6)[None, :]).astype(float)
        toks.append(iv("tok%d" % p, oh))
    counts = iv("counts", np.stack(toks, 1).sum(1))
    return counts
ALIGN = {"tok0": ["embed", "resid0.mid", "resid1.mid"],
         "tok1": ["embed", "resid0.mid", "resid1.mid"],
         "tok2": ["embed", "resid0.mid", "resid1.mid"],
         "tok3": ["embed", "resid0.mid", "resid1.mid"],
         "tok4": ["embed", "resid0.mid", "resid1.mid"],
         "tok5": ["embed", "resid0.mid", "resid1.mid"],
         "counts": ["attn0.out", "attn0.pattern", "attn1.out", "attn1.pattern"]}
