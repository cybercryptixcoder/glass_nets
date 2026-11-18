
import numpy as np

DESCRIPTION = ("Diagnostic. The network's task is the index of the SECOND LARGEST of the five inputs (88.9% agreement; "
               "disagreements are near-ties). Hypothesis being measured here: the hidden state is a linear function of "
               "the pairwise gaps relu(x_i - x_j) (and their opposites) accumulated over the prefix, i.e. the network "
               "works with comparisons of the inputs rather than with their signed magnitudes. Stage 1 is exact: "
               "h1 = x1*wx and wx has a fixed sign pattern, so h1 lies in the 2-D cone of the x1>0 ray and the x1<0 ray; "
               "u1 = (x1^+, x1^-). The output scores each position by its total gap below the running maximum, "
               "A_k = sum_j relu(x_j - x_k), picking the smallest positive A, which is the second largest position.")

PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def model(x, iv):
    u1 = iv("u1", np.stack([np.maximum(x[:, 0], 0), np.maximum(-x[:, 0], 0)], 1))
    x1 = u1[:, 0] - u1[:, 1]
    xx = np.concatenate([x1[:, None], x[:, 1:]], 1)

    v5 = None
    for t in range(1, 6):
        feats = []
        for (i, j) in PAIRS:
            if j < t:
                feats.append(np.maximum(xx[:, i] - xx[:, j], 0))
                feats.append(np.maximum(xx[:, j] - xx[:, i], 0))
        if t == 1:
            feats = [np.maximum(xx[:, 0], 0), np.maximum(-xx[:, 0], 0)]
        v = iv("g%d" % t, np.stack(feats, 1))
        if t == 5:
            v5 = v

    # A_k = sum over j != k of relu(x_j - x_k)
    A = np.zeros_like(xx)
    c = 0
    for (i, j) in PAIRS:
        gij = v5[:, c]
        gji = v5[:, c + 1]
        c += 2
        A[:, i] += gji   # relu(x_j - x_i)
        A[:, j] += gij   # relu(x_i - x_j)
    return -A - 1000.0 * (A < 1e-12)


ALIGN = {"u1": ["h1"], "g1": ["h1"], "g2": ["h2"], "g3": ["h3"], "g4": ["h4"], "g5": ["h5"]}
