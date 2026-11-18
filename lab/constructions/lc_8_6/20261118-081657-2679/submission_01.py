
import numpy as np
DESCRIPTION = "First stage only: the embedding site is the sum of a token embedding and a position embedding, so it is a linear function of the one-hot tokens at the 6 positions. Output is a placeholder (constant class 5)."
def model(x, iv):
    B = x.shape[0]
    toks = [iv("tok%d"%p, np.eye(6)[x[:,p]]) for p in range(6)]
    out = np.zeros((B,6)); out[:,5] = 1.0
    return out
ALIGN = {"tok0":["embed"],"tok1":["embed"],"tok2":["embed"],"tok3":["embed"],"tok4":["embed"],"tok5":["embed"]}
