
import numpy as np

DESCRIPTION = "One-layer ReLU RNN. At step 1 the state is proportional to either the positive or the negative lobe of W_x (2-d: x1+ and x1-). Each later step adds the next input through W_x plus the previous state through W_h, then ReLU. Output = h3 @ W_o. First pass: state coordinates carried explicitly (to be compressed)."

Wx = np.array([3.9653, 4.6995, -4.7163, -3.9652])
Wh = np.array([[1.3387, 1.9277, -2.3877, -1.8931],
               [-2.1996, -2.4339, 2.8057, 2.6429],
               [2.6352, 2.7868, -2.4339, -2.1912],
               [-1.8948, -2.3826, 1.9331, 1.3409]])
Wo = np.array([[-1.6718, -18.7477, 11.4305],
               [1.5993, 15.9993, -15.7403],
               [1.6175, 15.9592, -15.6973],
               [-1.6575, -18.7243, 11.4769]])
A = np.array([3.9653, 4.6995, 0.0, 0.0])   # h1 per unit of x1+
B = np.array([0.0, 0.0, 4.7163, 3.9652])   # h1 per unit of x1-

def model(x, iv):
    p = iv("pos", np.maximum(x[:, 0], 0.0))     # x1+
    n = iv("neg", np.maximum(-x[:, 0], 0.0))    # x1-
    h1 = p[:, None] * A + n[:, None] * B
    pre2 = x[:, 1:2] * Wx + h1 @ Wh
    h2 = iv("h2", np.maximum(pre2, 0.0))
    pre3 = x[:, 2:3] * Wx + h2 @ Wh
    h3 = iv("h3", np.maximum(pre3, 0.0))
    return h3 @ Wo

ALIGN = {"pos": ["h1"], "neg": ["h1"],
         "h2": ["h2"], "h3": ["h3"]}
