
import numpy as np
DESCRIPTION = "Diagnostic: probe how each site depends on the input tokens. Only the last-position token (5-dim one-hot) and the first-position token are named, at every site, to measure each site's intrinsic dimensionality and how much of it is token-linear. Output is still a placeholder constant class 5."
def model(x, iv):
    B = x.shape[0]
    t5 = iv("tok5", np.eye(6)[x[:, 5]][:, :5])
    t0 = iv("tok0", np.eye(6)[x[:, 0]][:, :5])
    out = np.zeros((B, 6)); out[:, 5] = 1.0
    return out
S = ["embed","attn0.pattern","attn0.out","resid0.mid","attn1.pattern","attn1.out","resid1.mid"]
ALIGN = {"tok5": S, "tok0": S}
