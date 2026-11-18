
import numpy as np

DESCRIPTION = """
The network compares the two numbers of the sequence: class 0 wins when the second
number exceeds a threshold set by the first, class 1 otherwise (the decision boundary
is essentially x2 = x1, with rectifier flat regions near it).
Step 1: x1 is split into its positive and negative parts by two rectifying units
  h1 = [4.174*relu(x1), 6.754*relu(-x1)]
  so the state records sign(x1) and |x1| in separate coordinates.
Step 2: those two coordinates shift the input of step 2, giving two competing evidence
units (uses x2 and the sign-dependent gain of x1):
  u0 = 4.174 * relu(x2 + 2.0852*relu(-x1) - 0.3821*relu(x1))   (evidence for class 0)
  u1 = 6.754 * relu(-x2 + 1.1415*relu(x1) - 0.7515*relu(-x1))  (evidence for class 1)
  Each unit is silent (0) when its evidence is negative, so a large separation in the
  'wrong' direction produces a tie-free but flat case (both units on/off patterns).
Readout: an antisymmetric pair of scores: class0 score = 9.541*u0 - 25.6923*u1,
class1 score = -9.5381*u0 + 25.6381*u1. Decision = larger score, i.e. class 0 iff
u0 > 2.69*u1.
"""

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    pos = iv("pos1", np.maximum(x1, 0.0))          # h1 unit 0 / 4.174
    neg = iv("neg1", np.maximum(-x1, 0.0))         # h1 unit 1 / 6.754
    u0 = iv("u0", 4.174 * np.maximum(x2 - 0.3821 * pos + 2.0852 * neg, 0.0))
    u1 = iv("u1", 6.754 * np.maximum(-x2 + 1.1415 * pos - 0.7515 * neg, 0.0))
    s0 = 9.541 * u0 - 25.6923 * u1
    s1 = -9.5381 * u0 + 25.6381 * u1
    return np.stack([s0, s1], axis=1)

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "u0": ["h2"], "u1": ["h2"]}
