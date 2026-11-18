
import numpy as np

DESCRIPTION = """
The network's task is a comparison of the two numbers of the sequence: class 0 wins
when the second number is larger than the first (class 1 when it is smaller).
Because it must do this with one rectifying unit per sign of the first number, it
proceeds in two steps.

Step 1 - split the first number by sign. The two units of h1 are ramps on +/-x1:
  h1 = [4.174*relu(x1), 6.754*relu(-x1)]
i.e. the state separates 'x1 positive' from 'x1 negative' and keeps the magnitude.
Only one of the two coordinates is ever nonzero.

Step 2 - compare x2 with a hurdle set by step 1. Each unit is a rectified difference
(evidence) and one unit serves each class:
  ev_above = relu(x2 - 0.3821*relu(x1) + 2.0852*relu(-x1))
      fires when x2 is above the hurdle; for x1>0 the hurdle is 0.382*x1 (just below
      the diagonal), for x1<0 it is -2.085*x1 (on the far side, so allowing x2 < x1
      of any size), hence ev_above is the 'x2 large' evidence.
  ev_below = relu(-x2 + 1.1415*relu(x1) - 0.7515*relu(-x1))
      the mirror image: fires when x2 is below 1.1415*x1 (x1>0) / -0.7515*x1 (x1<0),
      i.e. the 'x2 small' evidence.
So the sign of x1 selects which of two piecewise-linear hurdles is applied, and the
two rectifiers are silent (0) on the far side of the comparison.

Readout - an (almost) antisymmetric pair of scores over the two evidence values:
  score0 =  39.824*ev_above - 173.526*ev_below
  score1 = -39.812*ev_above + 173.160*ev_below
Decision = larger score, equivalent to ev_above > 4.353*ev_below, which reproduces
'x2 larger than x1 -> class 0'.
"""

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    pos = iv("pos1", np.maximum(x1, 0.0))        # = h1 unit 0 up to the gain 4.174
    neg = iv("neg1", np.maximum(-x1, 0.0))       # = h1 unit 1 up to the gain 6.754
    up = iv("ev_above", np.maximum(x2 - 0.3821 * pos + 2.0852 * neg, 0.0))
    dn = iv("ev_below", np.maximum(-x2 + 1.1415 * pos - 0.7515 * neg, 0.0))
    return np.stack([39.824 * up - 173.526 * dn,
                     -39.812 * up + 173.160 * dn], axis=1)

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "ev_above": ["h2"], "ev_below": ["h2"]}
