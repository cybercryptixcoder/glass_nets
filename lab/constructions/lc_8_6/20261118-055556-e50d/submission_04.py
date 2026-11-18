
import numpy as np

DESCRIPTION = ("The network reads 6 tokens from a vocabulary of 6 and returns 6 class scores; the decision is the argmax of "
               "position 5's residual (out_positions = [5]). Stage 1 (embedding) is established: the residual stream at the six "
               "positions is exactly the token embedding of that position's token plus a fixed position embedding, i.e. an affine "
               "function of the one-hot code of the token at each position, so one 6-dimensional one-hot variable per position "
               "reproduces the site exactly (explained 1.000, substitute 100%). The two attention layers are being reverse "
               "engineered; the provisional readout counts positions where the token index equals the position index (k), a feature "
               "the network's decisions are strongly correlated with (k=0 -> class 1, k>=1 -> class 0).")

def model(x, iv):
    x = np.asarray(x)
    E = np.eye(6)
    toks = [iv("tok%d" % p, E[x[:, p].astype(int)]) for p in range(6)]
    k = sum((x[:, p] == p).astype(float) for p in range(6))
    return np.stack([1.6 * k, 1.2 - 1.6 * k, 0.3 - 1.6 * k, -0.6 - 1.6 * k, -2.5 - 1.6 * k, -5.0 - 1.6 * k], 1)

ALIGN = {"tok%d" % p: ["embed"] for p in range(6)}
