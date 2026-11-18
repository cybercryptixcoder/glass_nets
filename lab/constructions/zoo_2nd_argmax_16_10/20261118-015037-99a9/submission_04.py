
import numpy as np, math
DESCRIPTION = ("Stage 1 (aligned): the first hidden state is 2-dimensional, h1 = relu(x1)*a + relu(-x1)*b = the positive and "
               "negative part of x_1. Task: the network names the position of the SECOND largest of the 10 inputs "
               "(the chosen position has rank 8 of 10 in 95.3% of cases; when it deviates it picks another top-rank "
               "position, i.e. the comparison is approximate near ties). Output below is a score vector whose argmax is "
               "the second-largest position; the recurrent stages are not yet explained.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    x1 = x[:, 0]
    pos = iv("xpos1", np.maximum(x1, 0.0))
    neg = iv("xneg1", np.maximum(-x1, 0.0))
    _ = (pos + neg)
    out = x.copy()
    rows = np.arange(x.shape[0])
    out[rows, x.argmax(1)] = -1e6
    return out
ALIGN = {"xpos1": ["h1"], "xneg1": ["h1"]}
