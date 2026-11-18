
import numpy as np

DESCRIPTION = ("The network is a 3-step ReLU recurrence with no biases. Each step adds the "
  "input's contribution (x_t * W_x, which is antisymmetric in the unit index: units 0,1 get "
  "positive weight, units 2,3 negative) to the previous state pushed through W_h, then rectifies. "
  "The state therefore keeps a running pair of 'signed gap' quantities (two units for the "
  "positive side, two for the negative side); each site's 4 activations are a fixed linear "
  "function of two scalars (per-step magnitude of the tracked positive/negative side). "
  "The final readout compares the two sides and picks the class = which of the three inputs is "
  "the median one (e.g. class 1 when x2 is between x1 and x3).")

Wx = np.array([3.9653, 4.6995, -4.7163, -3.9652])
Wh = np.array([[ 1.3387,  1.9277, -2.3877, -1.8931],
               [-2.1996, -2.4339,  2.8057,  2.6429],
               [ 2.6352,  2.7868, -2.4339, -2.1912],
               [-1.8948, -2.3826,  1.9331,  1.3409]])
Wo = np.array([[-1.6718, -18.7477,  11.4305],
               [ 1.5993,  15.9993, -15.7403],
               [ 1.6175,  15.9592, -15.6973],
               [-1.6575, -18.7243,  11.4769]])

def model(x, iv):
    B = x.shape[0]
    h = np.zeros((B, 4))
    for t in range(3):
        h = np.maximum(x[:, t:t+1] * Wx + h @ Wh, 0.0)
        h = iv("h%d" % (t + 1), h)
    return h @ Wo

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"]}
