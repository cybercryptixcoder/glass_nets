
import numpy as np

DESCRIPTION = ("The network reads a sequence of 6 tokens from a vocabulary of 6 and outputs 6 class scores. "
               "Stage 1 (embedding): the residual stream at the six positions is exactly the embedding of the token "
               "at that position plus a fixed position embedding, i.e. it is an affine function of the one-hot code of "
               "each token. We name one 6-dimensional one-hot variable per position (the token identity per position). "
               "Later stages (attention in layers 0 and 1 and the readout) are not yet explained; the returned scores "
               "are a provisional guess.")

def model(x, iv):
    x = np.asarray(x)
    E = np.eye(6)
    toks = [iv("tok%d" % p, E[x[:, p].astype(int)]) for p in range(6)]
    # provisional readout: nothing established yet
    return E[x.min(1).astype(int)]

ALIGN = {"tok%d" % p: ["embed"] for p in range(6)}
