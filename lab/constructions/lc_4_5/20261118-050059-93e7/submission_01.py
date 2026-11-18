
import numpy as np

DESCRIPTION = ("Diagnostic first step. The embedding site is the token embedding plus the "
               "position embedding: at each of the 5 positions the residual is W_E[token_p] + W_pos[p]. "
               "Nothing about the attention computation is known yet, so the output stage is a placeholder.")

WE = np.array([[-0.55,-2.03, 1.42, 0.30],
               [ 1.02,-1.09, 0.68, 1.85],
               [ 1.00,-0.20, 1.14, 1.40],
               [ 1.77, 1.90, 1.09,-0.52],
               [-0.64,-0.02, 2.88,-0.07]])
WP = np.array([[ 0.93,-0.53, 2.25, 0.03],
               [ 1.91,-1.15, 0.59,-1.67],
               [-1.30, 1.96, 2.55,-0.47],
               [-1.59,-0.45,-0.07, 2.06],
               [-1.57,-1.15, 0.17, 2.61]])

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    vs = []
    for p in range(5):
        oh = np.zeros((B, 5)); oh[np.arange(B), x[:, p]] = 1.0
        vs.append(iv("e%d" % p, oh @ WE + WP[p]))
    return np.zeros((B, 5))

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"],
         "e3": ["embed"], "e4": ["embed"]}
