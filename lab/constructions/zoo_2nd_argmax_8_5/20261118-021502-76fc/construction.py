
import numpy as np
DESCRIPTION = "One-layer ReLU RNN over 5 scalars. Stage 1 (exact): after x1 the 8-d state is rank 2, holding p = x1^+ along (0.42,0.045,0,0,0,0.64,0,7.00) and n = x1^- along (0,0,6.57,0,4.91,0,6.42,0). Later steps not yet modelled; readout placeholder = network's dominant class."
def model(x, iv):
    p = iv("p", np.maximum(x[:, 0], 0.0))
    n = iv("n", np.maximum(-x[:, 0], 0.0))
    out = np.zeros((x.shape[0], 5))
    out[:, 0] = 1.0
    return out
ALIGN = {"p": ["h1"], "n": ["h1"]}
