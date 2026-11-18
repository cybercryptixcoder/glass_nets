
import numpy as np
DESCRIPTION = ("Stage 1 only (partial construction). The four hidden units split by the sign of the first input: "
               "units 0 and 3 are driven positively (gains 0.9975 and 0.3939) and units 1 and 2 negatively "
               "(gains 2.2207 and 2.0354), so after step 1 the state is a rectified, sign-gated copy of x1: "
               "h1 = [0.9975*x1+, 2.2207*x1-, 2.0354*x1-, 0.3939*x1+]. Later stages are not yet explained; "
               "the output is a crude stand-in so the file runs.")
G = np.array([0.9975, 2.2207, 2.0354, 0.3939])
S = np.array([1.0, -1.0, -1.0, 1.0])

def model(x, iv):
    p = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    h1 = np.stack([G[0]*p, G[1]*n, G[2]*n, G[3]*p], 1)
    h1 = iv("h1_units", h1)
    return h1

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"], "h1_units": ["h1"]}
