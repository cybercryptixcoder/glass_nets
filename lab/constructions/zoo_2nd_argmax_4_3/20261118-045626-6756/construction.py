
import numpy as np

DESCRIPTION = ("The decision is which of the three inputs is the middle one (class 0/1/2 = x1/x2/x3 is the "
 "median). Read the state in the symmetric/antisymmetric coordinates of the unit-pair involution 0<->3, 1<->2: "
 "p=(h0+h3)/2, q=(h1+h2)/2, r=(h0-h3)/2, s=(h1-h2)/2. Three facts make the network tiny: "
 "(i) the input vector x_t*[A,B,-B,-A] lives purely in the antisymmetric pair (r,s): a step adds x_t*(A,B) there; "
 "(ii) W_h is block diagonal in these coordinates - the symmetric pair is mapped by a 2x2 matrix Ms with almost no "
 "gain (nearly rank one), the antisymmetric pair by a 2x2 matrix Ma with complex eigenvalues (a small rotation "
 "with gain ~2); "
 "(iii) the only coupling between the two blocks is the coordinatewise ReLU, and that is how magnitudes get into "
 "the symmetric part. So step 1 stores just two numbers, the magnitude |x1| and the signed value x1 (it is "
 "relu(x1)-relu(-x1) and relu(x1)+relu(-x1)), each a fixed 4-vector, and it feeds (p,q) = |x1|(A,B)/2, "
 "(r,s) = x1(A,B)/2. Steps 2 and 3 add the next input to (r,s), push the state through the two blocks and rectify, "
 "so the final symmetric pair (p,q) is a positive combination whose ratio q/p orders the three inputs. The readout "
 "sees only (p,q) of the last state and compares q/p with two thresholds (~1.17 and ~0.85): a large ratio means "
 "x2 is the median, a small one x3, the band between them x1. This also explains the network's own mistakes: they "
 "are inputs whose gaps are so small that the residual asymmetry of the weights (the ~1% non-block-diagonal part "
 "that this file drops) decides the comparison the other way.")

T = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 1., 0., -1.], [1., 0., -1., 0.]])
Ms = np.array([[-0.55416, -0.454704], [0.443644, 0.362320]])
Ma = np.array([[3.233713, 4.315592], [-4.834451, -5.230090]])
blk = np.zeros((4, 4)); blk[:2, :2] = Ms; blk[2:, 2:] = Ma
Wh = T @ blk @ T.T / 2
A1, B1, B2, A2 = 3.965263, 4.699488, 4.716320, 3.965202
Wx = np.array([A1, B1, -B2, -A2])
cp = np.array([-3.329386, -37.472013, 22.907393])
cq = np.array([3.216776, 31.958461, -31.437664])

def model(x, iv):
    def co(h):
        return ((h[:, 0] + h[:, 3]) / 2, (h[:, 1] + h[:, 2]) / 2,
                (h[:, 0] - h[:, 3]) / 2, (h[:, 1] - h[:, 2]) / 2)
    def og(p, q, r, s):
        return np.stack([p + r, q + s, q - s, p - r], 1)
    u = iv("u1", np.abs(x[:, 0]))
    v = iv("v1", x[:, 0])
    h = np.stack([A1 * (u + v) / 2, B1 * (u + v) / 2,
                  B2 * (u - v) / 2, A2 * (u - v) / 2], 1)
    p, q, r, s = co(h)
    for t, tag in ((1, "2"), (2, "3")):
        h = np.maximum(x[:, t:t + 1] * Wx + og(p, q, r, s) @ Wh, 0)
        p, q, r, s = co(h)
        p = iv("p" + tag, p); q = iv("q" + tag, q)
        r = iv("r" + tag, r); s = iv("s" + tag, s)
    return p[:, None] * cp + q[:, None] * cq

ALIGN = {"u1": ["h1"], "v1": ["h1"],
         "p2": ["h2"], "q2": ["h2"], "r2": ["h2"], "s2": ["h2"],
         "p3": ["h3"], "q3": ["h3"], "r3": ["h3"], "s3": ["h3"]}
