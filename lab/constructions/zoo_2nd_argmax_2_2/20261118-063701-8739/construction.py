
import numpy as np
DESCRIPTION = ("The network is a comparator: class 0 iff x2 > x1 (the stored value against the new input). "
 "Stage 1 (h1) splits x1 into its positive part p = x1^+ and its negative part n = x1^-; the two units "
 "are scaled copies of p and n. Stage 2 (h2) builds two nonnegative gates from the input pair, "
 "gate0 = relu(x2 - a*p + b*n) and gate1 = relu(-x2 + c*p + d*n). Each gate is positive on one side of a "
 "threshold line in the (x1,x2) plane and the two lines bracket the diagonal x2 = x1: outside the band "
 "exactly one gate is on, which already answers the question, and inside the band the readout compares "
 "them with weight k. k is not free: it is fixed by the two lines meeting on the diagonal, "
 "k*(1+a) = c-1 and k*(b-1) = 1+d, so the antisymmetric score k*gate0 - gate1 vanishes exactly on "
 "x2 = x1 and its sign is the comparison of the two numbers.")
def model(x, iv):
    x1 = x[:, 0]; x2 = x[:, 1]
    p = iv("p", np.maximum(x1, 0))
    n = iv("n", np.maximum(-x1, 0))
    A = -0.3821     # gate0 threshold slope on p
    B = 2.0849      # gate0 threshold slope on n
    C = 1.1415      # gate1 threshold slope on p
    D = -0.7515     # gate1 threshold slope on n
    a0 = iv("a0", np.maximum(x2 + A*p + B*n, 0))
    a1 = iv("a1", np.maximum(-x2 + C*p + D*n, 0))
    k = (1 + D)/(B - 1)     # readout weight: makes both gates vanish on x2 = x1
    s = k*a0 - a1
    return np.stack([s, -s], 1)
ALIGN = {"p": ["h1"], "n": ["h1"], "a0": ["h2"], "a1": ["h2"]}
