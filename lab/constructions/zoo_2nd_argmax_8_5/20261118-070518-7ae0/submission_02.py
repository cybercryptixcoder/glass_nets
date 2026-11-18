
import numpy as np

DESCRIPTION = ("Step 1 is explained: h1 = relu(x1)*a+ + relu(-x1)*a-, i.e. the first hidden state is a "
               "positive mixture of just two fixed rays, one for each sign of the first input (positive "
               "branch gains [0.64,0.42,0.05,7.00] on units 0,2,4,7; negative branch gains [6.42,6.57,4.91,0.05] "
               "on units 1,3,5,6). So the state after step 1 carries exactly 2 numbers: the sizes of the "
               "positive and negative parts of x1. Later steps here are still the raw recurrence (to compress).")

RP = np.array([0.6426,0.0,0.4235,0.0,0.0451,0.0,0.0,7.0017])
RN = np.array([0.0,6.4213,0.0,6.5737,0.0,4.9132,0.0541,0.0])
WX = RP - RN
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
