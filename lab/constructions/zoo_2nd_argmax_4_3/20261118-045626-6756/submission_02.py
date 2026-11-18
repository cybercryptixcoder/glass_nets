
import numpy as np

DESCRIPTION = ("Three-step ReLU recurrence on 3 numbers whose decision is which input is the MIDDLE one "
  "(class 0 = x1 median, 1 = x2 median, 2 = x3 median). The state is written in the symmetric/antisymmetric "
  "coordinates of the unit-pair involution (0<->3, 1<->2): p=(h0+h3)/2, q=(h1+h2)/2, r=(h0-h3)/2, s=(h1-h2)/2. "
  "In those coordinates the input vector x_t*[A,B,-B,-A] is purely antisymmetric (it adds x_t*(A,B) to (r,s)), "
  "and W_h is nearly block diagonal: it maps the symmetric pair (p,q) by a nearly rank-1 2x2 matrix and the "
  "antisymmetric pair (r,s) by a 2x2 matrix with complex eigenvalues (a small rotation-amplifier). The two blocks "
  "are coupled only by the coordinatewise ReLU, which is what lets the input magnitude enter the symmetric part. "
  "The readout uses only the symmetric pair (p,q) of the final state; its decision boundary is the ratio q/p "
  "crossing two thresholds, which is what makes the middle input select the class.")

T = np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[0.,1.,0.,-1.],[1.,0.,-1.,0.]])
Ms = np.array([[-0.55416, -0.45470], [0.44364, 0.36232]])
Ma = np.array([[3.23371, 4.31559], [-4.83445, -5.23009]])
blk = np.zeros((4, 4)); blk[:2, :2] = Ms; blk[2:, 2:] = Ma
Wh = T.T @ blk @ T / 2.0
Wx = np.array([3.965232, 4.69949, -4.71632, -3.965232])
cp = np.array([-0.10418, -1.17251, 0.71679])
cq = np.array([0.10065, 1.0, -0.98370])

def model(x, iv):
    B = x.shape[0]
    h = np.zeros((B, 4))
    for t in range(3):
        h = np.maximum(x[:, t:t+1] * Wx + h @ Wh, 0.0)
        h = iv("h%d" % (t + 1), h)
    p = iv("p", (h[:, 0] + h[:, 3]) / 2.0)
    q = iv("q", (h[:, 1] + h[:, 2]) / 2.0)
    return p[:, None] * cp + q[:, None] * cq

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3", "p", "q"]}
