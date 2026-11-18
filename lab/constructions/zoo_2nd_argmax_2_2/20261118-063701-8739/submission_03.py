
import numpy as np
DESCRIPTION = ("The network decides 'is x2 bigger than x1?': class 0 iff x2 > x1. "
 "Stage 1 (h1) splits x1 into its positive part p = x1^+ and negative part n = x1^-; the two hidden "
 "units are just scaled copies of p and n (4.174 and 6.754), scale is irrelevant. "
 "Stage 2 (h2) forms two nonnegative gates, a0 = relu(x2 - 0.3821 p + 2.0849 n) and "
 "a1 = relu(-x2 + 1.1415 p - 0.7515 n). a0 is positive above a threshold line in x1, a1 is positive "
 "below another threshold line; the two lines bracket x2 = x1. Outside that band exactly one gate is on "
 "(giving the right answer outright), inside the band the readout compares them with weight k, and k is "
 "fixed by the requirement that both gates flip sign on x2 = x1: k(1-0.3821) = 1.1415-1 and "
 "k(2.0849-1) = 1-0.7515. So the readout antisymmetric score k*a0 - a1 crosses zero exactly at x2 = x1, "
 "i.e. the network is a comparator between the stored value and the new input.")
def model(x, iv):
    x1 = x[:, 0]; x2 = x[:, 1]
    p = iv("p", np.maximum(x1, 0))
    n = iv("n", np.maximum(-x1, 0))
    a0 = iv("a0", np.maximum(x2 - 0.3821*p + 2.0849*n, 0))
    a1 = iv("a1", np.maximum(-x2 + 1.1415*p - 0.7515*n, 0))
    k = (1 - 0.7515)/(2.0849 - 1)          # = 0.22905, readout weight for the a0 gate
    s = k*a0 - a1
    return np.stack([s, -s], 1)
ALIGN = {"p": ["h1"], "n": ["h1"], "a0": ["h2"], "a1": ["h2"]}
