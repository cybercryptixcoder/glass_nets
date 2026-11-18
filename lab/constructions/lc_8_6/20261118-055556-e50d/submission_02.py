
import numpy as np

DESCRIPTION = ("The network reads 6 tokens from a vocabulary of 6 and returns 6 class scores; the decision is the argmax. "
               "Stage 1 (embedding) is solved: the residual stream at the six positions is exactly the token embedding of "
               "that position's token plus a fixed position embedding, i.e. an affine function of the one-hot code of the "
               "token in each position. So we name one 6-dimensional one-hot variable per position; together they reproduce "
               "the embedding site exactly. The attention stages are still being worked out; the provisional readout below "
               "counts the positions where the token index equals the position index (a feature the network's decisions are "
               "strongly correlated with) and maps that count to class scores.")

def model(x, iv):
    x = np.asarray(x)
    E = np.eye(6)
    toks = [iv("tok%d" % p, E[x[:, p].astype(int)]) for p in range(6)]
    B = x.shape[0]
    idx = np.arange(6)
    k = sum((x[:, p] == p).astype(float) for p in range(6))
    # provisional: a monotone map of the diagonal-match count
    return np.stack([k, 1.5 - k, 0.5 - 0.8 * k, -3.0 - 0.2 * k, -6.0, -8.0]) @ np.eye(6) if False else \
           np.stack([k, 1.0 - k, 0.4 - 0.7 * k, -2.5 - 0.1 * k, -5.0 + 0 * k, -7.0 + 0 * k], 1)

ALIGN = {"tok%d" % p: ["embed"] for p in range(6)}
