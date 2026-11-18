
import numpy as np

DESCRIPTION = ("ROUGH first draft. Stage 1: the hidden state after the first input is exactly "
               "relu(x1 * W_x), with the 8 measured input weights (5 of them nearly zero). "
               "The readout is not yet explained: the network's decision is close to the index of "
               "the largest input value (33% exact, 95% within top-3), so model() currently ranks "
               "classes by x itself.")

WX = np.array([-0.207, 0.030, -1.203, -0.151, -0.467, 0.003, 0.619, 0.004])

def model(x, iv):
    a1 = iv("a1", np.maximum(x[:, 0:1] * WX[None, :], 0.0))
    # placeholder readout: score each class by its input value
    return x

ALIGN = {"a1": ["h1"]}
