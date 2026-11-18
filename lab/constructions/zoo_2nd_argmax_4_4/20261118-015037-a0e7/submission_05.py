
import numpy as np
DESCRIPTION = ("Four rectified accumulators, one update per input, with the units' activities measured in units of "
               "their own input gain. Step 1 is pure gating of the first input: units 0 and 3 take +x1, units 1 and 2 "
               "take -x1, so the state after step 1 is exactly [x1+, x1-, x1-, x1-] - only two numbers, the positive "
               "and the negative part of x1. Every later step adds one signed copy of x_t to every unit (units 0,3 "
               "read +x_t, units 1,2 read -x_t), mixes the old state through a dense recurrent gain matrix A, and "
               "rectifies: s_t = relu(x_t*S + s_(t-1) @ A). A amplifies (dominant eigenvalue 2.56), so activity "
               "grows from step to step and each new input is read on top of a large running state; the rectification "
               "is what makes the network forget negative accumulations. The readout compares the final state with "
               "three fixed contrast directions (class 1, 2, 3 against class 0) whose own score is the zero "
               "baseline, so the decision is the argmax of [0, c1.s4, c2.s4, c3.s4].")
S = np.array([1.0, -1.0, -1.0, 1.0])            # signed unit input gain per unit (normalized to +-1)
A = np.array([                                   # recurrent gain matrix (from unit i to unit j)
    [-0.213737, 1.273340, -0.069905, 2.318979],
    [5.417269, 1.320339, -0.533874, 14.224694],
    [-4.427177, -1.289346, 0.530658, -14.462238],
    [-0.013221, 0.084671, 0.482565, -0.752541]])
C = np.array([                                   # contrast directions: class 1, 2, 3 against class 0
    [-1.005312, -4.188698, -1.088185],
    [-3.579383, -4.633421, -0.978169],
    [4.911669, 4.261590, 0.563860],
    [0.793057, 2.319706, 0.658305]])

def model(x, iv):
    p = iv("x1_pos", np.maximum(x[:, 0], np.zeros_like(x[:, 0])))    # positive part of the first input
    n = iv("x1_neg", np.maximum(-x[:, 0], np.zeros_like(x[:, 0])))   # negative part of the first input
    s = np.stack([p, n, n, p], 1)                                    # step-1 state: 4 units, 2 numbers
    for t in range(1, x.shape[1]):
        drive = x[:, t:t+1] * S + s @ A          # add the new input with its signed unit gain, plus mixed state
        s = iv("s%d" % t, np.maximum(drive, np.zeros_like(drive)))
    cont = s @ C                                 # class j - class 0 for j = 1, 2, 3
    return np.concatenate([np.zeros_like(cont[:, :1]), cont], 1)

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"], "s1": ["h2"], "s2": ["h3"], "s3": ["h4"]}
