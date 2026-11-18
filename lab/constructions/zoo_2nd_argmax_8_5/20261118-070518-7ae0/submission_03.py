
import numpy as np

DESCRIPTION = ("Re-implementation of the RNN written as a ray recursion, used to locate the mechanism. Step 1 is "
               "exact and low dimensional: h1 = relu(x1)*a+ + relu(-x1)*a- (two rays, so the state after step 1 "
               "carries only the positive and negative parts of x1). The pre-activation of step t>1 is a linear "
               "function of (x_t, h_{t-1}); later steps are still the raw weights pending compression.")

RP = np.array([0.6426,0.,0.4235,0.,0.0451,0.,0.,7.0017])
RN = np.array([0.,6.4213,0.,6.5737,0.,4.9132,0.0541,0.])
WX = RP - RN
C = RP @ None if False else None
WH = np.array([
[ 0.125, 0.155, 0.476, 0.061, 0.771, 0.036, 0.944,-0.163],
[-0.465,-0.236, 0.272, 0.493, 0.169,-0.732, 0.424,-0.027],
[ 0.020,-0.129,-0.107,-0.190,-0.618,-0.424, 0.240, 0.082],
[ 0.528,-0.300,-0.363,-0.847, 0.279, 0.570, 0.199, 0.061],
[-0.177,-0.324, 0.216,-0.252,-0.747,-0.192,-1.207, 0.380],
[ 0.881, 0.600, 0.126, 0.348, 0.127, 0.070,-0.750, 0.024],
[ 0.127, 0.962,-0.086, 0.973, 0.546, 0.680, 0.787,-0.890],
[-1.656,-0.017, 1.419, 1.088, 0.156, 0.815, 1.102,-0.828]])
WO = np.array([
[ 0.716,-1.137, 0.680,-1.948, 0.727],
[ 0.336,-0.716,-3.905, 3.885,-0.239],
[ 0.849,-0.069,-3.387, 0.567, 0.633],
[ 0.360,-2.426, 4.804,-2.857,-0.673],
[-0.403, 2.160,-0.467,-0.468,-0.377],
[-0.871, 4.200,-1.351,-1.310,-0.845],
[-0.867,-0.938, 0.518, 0.679, 0.493],
[ 0.240, 0.317, 0.378, 0.356,-2.253]])

def model(x, iv):
    p = iv("pos1", np.maximum(x[:, 0], 0.0))
    n = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h = p[:, None] * RP + n[:, None] * RN
    for t in range(2, 6):
        h = iv("h%d" % t, np.maximum(x[:, t-1:t] * WX + h @ WH, 0.0))
    return h @ WO

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "h5": ["h5"]}
