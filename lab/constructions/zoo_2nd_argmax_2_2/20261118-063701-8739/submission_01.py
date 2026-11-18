
import numpy as np
DESCRIPTION = ("The network compares the two inputs. Step 1 stores the positive and negative parts of x1 "
 "in the two hidden units (unit0 = 4.174*x1^+, unit1 = 6.754*x1^-). Step 2 computes "
 "a0 = relu(x2 - 0.3821*p + 2.0849*n) and a1 = relu(-x2 + 1.1415*p - 0.7515*n); the readout is "
 "antisymmetric (s = 0.2297*a0 - a1), so class 0 iff x2 > x1.")
def model(x, iv):
    x1 = x[:,0]; x2 = x[:,1]
    p = iv("p", np.maximum(x1, 0))
    n = iv("n", np.maximum(-x1, 0))
    a0 = iv("a0", np.maximum(x2 - 0.3821*p + 2.0849*n, 0))
    a1 = iv("a1", np.maximum(-x2 + 1.1415*p - 0.7515*n, 0))
    s = 0.22975*a0 - a1
    return np.stack([s, np.zeros_like(s)], 1)
ALIGN = {"p": ["h1"], "n": ["h1"], "a0": ["h2"], "a1": ["h2"]}
