
import numpy as np

DESCRIPTION = ("The task is 'which of the three inputs is the middle one' (class 0/1/2 = x1/x2/x3 is the median). "
 "The recurrence is best read in the symmetric/antisymmetric coordinates of the unit-pair involution "
 "0<->3, 1<->2:  p=(h0+h3)/2, q=(h1+h2)/2, r=(h0-h3)/2, s=(h1-h2)/2.  In these coordinates "
 "(i) the input vector x_t*[A,B,-B,-A] is purely antisymmetric, so x_t only ever enters the (r,s) pair "
 "(it adds x_t*(A,B) there); "
 "(ii) W_h is block diagonal: the symmetric pair (p,q) is mapped by a 2x2 matrix Ms that is nearly rank one "
 "and has almost no gain (it shrinks by ~0.2), while the antisymmetric pair (r,s) is mapped by a 2x2 matrix Ma "
 "with complex eigenvalues (a small rotation with gain ~2); "
 "(iii) the only thing that couples the two blocks is the coordinatewise ReLU, and that is how the magnitudes of "
 "the inputs get into the symmetric part. "
 "So step 1 fills (r,s) with x1*(A,B) and the ReLU also lets its magnitude into (p,q) = |x1|*(A/2,B/2); each later "
 "step mixes the previous state through the two blocks and rectifies, which keeps a signed running comparison "
 "between the current input and the previously accumulated magnitude; after three steps the symmetric pair (p,q) "
 "holds a positive combination whose ratio q/p orders the three inputs, and the readout, which only sees (p,q), "
 "compares that ratio with two thresholds (1.17 and 0.85): a large ratio means x2 is the median, a small one x3, "
 "and the intermediate band x1. Near-ties are decided by the residual coupling/scale asymmetries of the weights, "
 "which is why the network itself is occasionally wrong.")

T = np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[0.,1.,0.,-1.],[1.,0.,-1.,0.]])
Ms = np.array([[-0.55416, -0.454704], [0.443644, 0.362320]])
Ma = np.array([[3.233713, 4.315592], [-4.834451, -5.230090]])
blk = np.zeros((4, 4)); blk[:2, :2] = Ms; blk[2:, 2:] = Ma
Wh = T @ blk @ T.T / 2
A, B1, B2 = 3.965232, 4.699488, 4.716320
Wx = np.array([A, B1, -B2, -A])
cp = np.array([-3.329386, -37.472013, 22.907393])
cq = np.array([3.216776, 31.958461, -31.437664])

def model(x, iv):
    def to_orig(p, q, r, s):
        return np.stack([p + r, q + s, q - s, p - r], 1)
    def to_coords(h):
        return ((h[:, 0] + h[:, 3]) / 2, (h[:, 1] + h[:, 2]) / 2,
                (h[:, 0] - h[:, 3]) / 2, (h[:, 1] - h[:, 2]) / 2)
    h = np.maximum(x[:, 0:1] * Wx, 0)
    p, q, r, s = to_coords(h)
    p = iv("p1", p); q = iv("q1", q); r = iv("r1", r); s = iv("s1", s)
    for t, tag in ((1, "2"), (2, "3")):
        h = np.maximum(x[:, t:t + 1] * Wx + to_orig(p, q, r, s) @ Wh, 0)
        p, q, r, s = to_coords(h)
        p = iv("p" + tag, p); q = iv("q" + tag, q)
        r = iv("r" + tag, r); s = iv("s" + tag, s)
    return p[:, None] * cp + q[:, None] * cq

ALIGN = {"p1": ["h1"], "q1": ["h1"], "r1": ["h1"], "s1": ["h1"],
         "p2": ["h2"], "q2": ["h2"], "r2": ["h2"], "s2": ["h2"],
         "p3": ["h3"], "q3": ["h3"], "r3": ["h3"], "s3": ["h3"]}
