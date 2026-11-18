
import numpy as np
DESCRIPTION = ("The network compares the two inputs: class 0 iff x2 > x1. Step 1 splits x1 into its "
 "positive part p=x1^+ and negative part n=x1^- (unit0 = 0.7642*p, unit1 = 2.0849*n). Step 2 forms two "
 "nonnegative gates: a0 = relu(x2 - 0.5*p + n) and a1 = relu(-x2 + 1.4937*p - 0.36045*n). Their ReLUs are "
 "off precisely outside the band that contains the line x2=x1, so the sign of (0.2297*a0 - a1) is the sign "
 "of x2 - x1: the stored value competes with the new input.")
def model(x, iv):
    x1 = x[:, 0]; x2 = x[:, 1]
    p = iv("p", 0.7642*np.maximum(x1, 0))
    n = iv("n", 2.0849*np.maximum(-x1, 0))
    a0 = iv("a0", np.maximum(x2 - 0.5*p + n, 0))
    a1 = iv("a1", np.maximum(-x2 + 1.4937*p - 0.36045*n, 0))
    s = 0.2297*a0 - a1
    return np.stack([s, np.zeros_like(s)], 1)
ALIGN = {"p": ["h1"], "n": ["h1"], "a0": ["h2"], "a1": ["h2"]}
