
import numpy as np
DESCRIPTION = "One-layer ReLU RNN over 5 scalars. Stage 1 (exact): after x1 the 8-d state is rank 2: unit 7 carries 7.00*x1^+ and units 0,1,5 carry proportional fractions of x1^+ (0.42,0.045,0.64), while units 2,4,6 carry 6.57,4.91,6.42 times x1^-. So the state is p = x1^+ and n = x1^- in fixed directions. Later stages: pre-activations z_t = x_t*W_x + h_{t-1}*W_h occupy 3 dims at t=2 and 8 at later steps; not yet modelled, so the readout is the network's dominant class."
def model(x, iv):
    p = iv("p", np.maximum(x[:, 0], 0.0))
    n = iv("n", np.maximum(-x[:, 0], 0.0))
    out = np.zeros((x.shape[0], 5))
    out[:, 0] = 1.0
    return out
ALIGN = {"p": ["h1"], "n": ["h1"]}
