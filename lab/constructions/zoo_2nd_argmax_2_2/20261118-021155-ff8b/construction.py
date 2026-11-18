
import numpy as np

DESCRIPTION = """
The recurrence keeps a two-unit "which side of zero is x1" detector: h1 = [relu(x1), relu(-x1)] up to the
two input gains, so exactly one of the two units is alive and it records |x1|.
Step 2 compares the second input x2 against a threshold that depends on that detector, and reports the
amount by which x2 is above / below it (that surplus is what the second state holds):
   u1 (above) = relu( x2 + 1.2886*h1p + 2.9888*h1n )
   u2 (below) = relu( -(x2 + 0.2362*h1p + 0.7515*h1n) )
i.e. when x1>0 the upper threshold is -1.2886*x1 and the lower one -0.2362*x1; when x1<0 they are
-2.9888*|x1| and -0.7515*|x1|. So the upper and lower thresholds are both increasing in x1 (a factor ~5.5
between them) and the two surpluses are mutually exclusive for x1>0, partly coexisting for x1<0.
The output is an almost perfectly antisymmetric readout: class-0 score = u1 - 0.6020*u2 (class 1 = minus
that). Hence the decision is "x2 above its threshold" (class 0) versus "x2 below it" (class 1), with the
surplus of the other unit weighted 0.60.
"""

K_UP_POS = 1.2886474133    # upper threshold slope, x1 > 0   (threshold = -K*x1)
K_UP_NEG = 2.9888235050    # upper threshold slope, x1 < 0
K_DN_POS = 0.2361550500    # lower threshold slope, x1 > 0
K_DN_NEG = 0.7515226603    # lower threshold slope, x1 < 0
W_DN = 0.6019845105        # readout weight of the lower surplus relative to the upper one

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    # step 1: the two mutually exclusive detectors of x1
    h1p = iv("h1_pos", np.maximum(x1, 0))     # alive iff x1 > 0
    h1n = iv("h1_neg", np.maximum(-x1, 0))    # alive iff x1 < 0
    # step 2: how far x2 lies above the upper threshold, and below the lower one
    upper = iv("h2_above", np.maximum(x2 + K_UP_POS * h1p + K_UP_NEG * h1n, 0))
    lower = iv("h2_below", np.maximum(-(x2 + K_DN_POS * h1p + K_DN_NEG * h1n), 0))
    # output: class 0 = x2 above its threshold, class 1 = the mirror image
    s0 = upper - W_DN * lower
    return np.stack([s0, -s0], axis=1)

ALIGN = {"h1_pos": ["h1"], "h1_neg": ["h1"], "h2_above": ["h2"], "h2_below": ["h2"]}
