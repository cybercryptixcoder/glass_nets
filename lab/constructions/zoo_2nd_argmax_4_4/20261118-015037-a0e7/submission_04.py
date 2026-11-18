
import numpy as np
DESCRIPTION = ("Four rectified accumulators, one step per input. Each unit's activity is measured in units of its own "
               "input gain, so a new input pushes every unit by exactly one signed copy of x_t: units 0 and 3 read "
               "+x_t, units 1 and 2 read -x_t (this is why after step 1 the state is just [x1+, x1-, x1-, x1+]). "
               "Between inputs the state is mixed by a dense recurrent gain matrix A and rectified, so the state is a "
               "gated accumulation of the four inputs: s_t = relu(x_t*S + s_(t-1) @ A). A amplifies (its dominant "
               "eigenvalue is 2.56), so activity grows step over step and later inputs are read on top of a large "
               "running state. The readout scores the classes by comparing the final state with three fixed "
               "contrast directions (class1, class2, class3 against class0); class0's score is the zero baseline, "
               "so the decision is the argmax of [0, c1.s4, c2.s4, c3.s4].")
S = np.array([1.0, -1.0, -1.0, 1.0])          # input gain per unit (normalized to +-1)
A = np.array([                                 # recurrent gain matrix (unit i -> unit j)
    [-0.213737, 1.273340, -0.069905, 2.318979],
    [5.417269, 1.320339, -0.533874, 14.224694],
    [-4.427177, -1.289346, 0.530658, -14.462238],
    [-0.013221, 0.084671, 0.482565, -0.752541]])
C = np.array([                                 # contrast directions of class 1,2,3 against class 0
    [-1.005312, -4.188698, -1.088185],
    [-3.579383, -4.633421, -0.978169],
    [4.911669, 4.261590, 0.563860],
    [0.793057, 2.319706, 0.658305]])

def model(x, iv):
    s = np.zeros_like(x)                       # state: one accumulator per unit, starts at rest
    for t in range(x.shape[1]):
        drive = x[:, t:t+1] * S + s @ A        # new input added with its signed unit gain, plus mixed old state
        s = iv("s%d" % t, np.maximum(drive, np.zeros_like(drive)))
    cont = s @ C                               # class j=1,2,3 minus class 0
    return np.concatenate([np.zeros_like(cont[:, :1]), cont], 1)

ALIGN = {"s0": ["h1"], "s1": ["h2"], "s2": ["h3"], "s3": ["h4"]}
