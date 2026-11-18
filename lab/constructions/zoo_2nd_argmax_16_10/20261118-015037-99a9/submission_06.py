
import numpy as np, math
DESCRIPTION = ("Stage 1 (aligned): h1 = relu(x1)*a + relu(-x1)*b, so after one step the state is a 2-dimensional encoding of "
               "the positive and negative part of x_1. Task: the network names the position of the SECOND largest of the 10 "
               "inputs (chosen position has rank 8 of 10 in 95.3% of cases). Probing the pre-activations shows each large-|Wx| "
               "unit compares x_t against a running level close to the prefix maximum: units 1,6,7,8 = relu(c - |Wx| x_t) and "
               "unit 4 = relu(Wx (x_t - c')) with c ~ max of the prefix (small leak), i.e. the recurrence keeps a running "
               "comparison level (and the answer position) rather than an exact order statistic, which is why it errs at "
               "near-ties. Recurrent stages not yet fully explained; output below is the second-largest rule.")
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
