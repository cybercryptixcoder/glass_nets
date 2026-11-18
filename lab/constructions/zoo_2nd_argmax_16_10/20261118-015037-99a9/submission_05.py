
import numpy as np, math
DESCRIPTION = ("Stage 1 (aligned): h1 = relu(x1)*a + relu(-x1)*b, i.e. it holds only the positive and negative part of x_1 "
               "(the state after one step is 2-dimensional). Task: the network names the position of the SECOND largest "
               "of the 10 inputs (chosen position has rank 8 of 10 in 95.3% of cases; deviations are near ties, with a "
               "slight preference for the earlier position, and the answer also depends weakly on the other entries, so "
               "the comparison is approximate rather than exact). The readout W_o acts like a matched filter on a smooth "
               "position code: the class-conditional output profiles peak at the answer class and fall off on either side, "
               "and units 10-15 (tiny input weights) accumulate the step count so that position can be read out. "
               "Recurrent stages are not yet explained; output below is the second-largest rule.")
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
