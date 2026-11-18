
import numpy as np
DESCRIPTION = "One-layer ReLU RNN. Stage 1: after reading x1 the hidden state has exactly rank 2 structure: it holds the positive part x1^+ and the negative part x1^- (magnitude), each fed into separate groups of units (unit 7 for x1^+; units 2,4,6 for x1^-). Later stages not yet explained, so the readout is a placeholder."
def model(x, iv):
    p = iv("p", np.maximum(x[:, 0], 0.0))
    n = iv("n", np.maximum(-x[:, 0], 0.0))
    # placeholder readout: network overwhelmingly picks class 0
    out = np.zeros((x.shape[0], 5))
    out[:, 0] = 1.0
    return out
ALIGN = {"p": ["h1"], "n": ["h1"]}
