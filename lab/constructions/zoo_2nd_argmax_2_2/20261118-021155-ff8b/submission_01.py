
import numpy as np

DESCRIPTION = """
Two-step ReLU recurrence on a sequence (x1, x2). The input weight vector has opposite signs on the two
hidden units, so h1 is a pair of mutually exclusive "which side of zero is x1" detectors:
  h1 = [ relu(x1)*4.174 , relu(-x1)*6.754 ]   (exactly one unit nonzero)
The second step adds the same input vector scaled by x2 to a linear readout of h1, then rectifies:
  h2 = relu( x2*[4.174, -6.754] + h1 @ [[1.2886, -0.3821],[1.8471, -0.7515]] )
i.e. h2_1 = relu(4.174*x2 + 1.2886*h1_pos + 1.8471*h1_neg), h2_2 = relu(-6.754*x2 - 0.3821*h1_pos - 0.7515*h1_neg).
The output is a nearly rank-1 linear readout: class-0 score = 25.638*h2_1 - 9.538*h2_2 and class 1 is its
mirror image (class-1 score = -(25.692*h2_1 - 9.541*h2_2)). Decision = argmax = sign of the class-0 score.
"""

A, Bg = 4.1740379333, 6.7540297508          # input gains, one per hidden unit (opposite signs)
R11, R12 = 1.2886474133, -0.3821235597      # recurrent row: from h1 unit 1
R21, R22 = 1.8471139669, -0.7515226603      # recurrent row: from h1 unit 2
O01, O02 = 25.6380691528, -9.5381479263     # class-0 readout of (h2_1, h2_2)
O11, O12 = -25.6923103333, 9.5409688950     # class-1 readout (mirror image)

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    # step 1: two mutually exclusive sign detectors of x1
    hp = iv("h1_pos", np.maximum(x1, 0.0) * A)    # h1 unit 1: fires when x1 > 0
    hn = iv("h1_neg", np.maximum(-x1, 0.0) * Bg)   # h1 unit 2: fires when x1 < 0
    # step 2: accumulate x2 through the input weights plus a linear mix of h1, then rectify
    p1 = x2 * A + hp * R11 + hn * R21
    p2 = -x2 * Bg + hp * R12 + hn * R22
    u1 = iv("h2_1", np.maximum(p1, 0.0))
    u2 = iv("h2_2", np.maximum(p2, 0.0))
    # readout
    s0 = u1 * O01 + u2 * O02
    s1 = u1 * O11 + u2 * O12
    return np.stack([s0, s1], axis=1)

ALIGN = {"h1_pos": ["h1"], "h1_neg": ["h1"], "h2_1": ["h2"], "h2_2": ["h2"]}
