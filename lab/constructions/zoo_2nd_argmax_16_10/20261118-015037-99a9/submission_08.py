
import numpy as np, math
DESCRIPTION = ("Stage 1: the first hidden state is 2-dimensional, h1 = relu(x1)*a + relu(-x1)*b, i.e. it holds only the positive "
               "and negative part of x_1 (so the state after one step is a faithful encoding of x_1). The network's answer is the "
               "position of the SECOND largest input (rank 8 of 10 in 95.3% of cases), computed from a running comparison level; "
               "its deviations from an exact second-max occur only at near ties (|M2-M3|<0.1, |M1-M2|<0.2), i.e. the comparison "
               "is smoothed/blurred, and the state carries the whole prefix in a smooth position code (the readout acts like a "
               "matched filter over the answer position). The model reconstructs x_1 from the two stage-1 variables and applies "
               "the second-maximum readout to the reconstructed sequence.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    pos = iv("xpos1", np.maximum(x[:, 0], 0.0))
    neg = iv("xneg1", np.maximum(-x[:, 0], 0.0))
    xe = x.copy()
    xe[:, 0] = pos - neg
    out = xe.copy()
    rows = np.arange(xe.shape[0])
    out[rows, xe.argmax(1)] = -1e6
    return out
ALIGN = {"xpos1": ["h1"], "xneg1": ["h1"]}
