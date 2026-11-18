
import numpy as np

DESCRIPTION = ("Task: the runner-up. The winning class is the position of the SECOND largest of the 5 inputs "
               "(agrees with argmax of the 2nd largest value on ~88% of inputs; disagreements are close values; "
               "the network is graded wrong on ~11%). Diagnostics that guide the mechanism: (i) the per-position "
               "output score is essentially a function of the RANK of that input inside the sequence -- mean score "
               "by rank is -12.5, -6.6, -1.2, +5.5, -6.2 for rank 0(min) .. rank 4(max) -- so the state must carry "
               "rank information for every position; (ii) stage 1 is settled: h1 = relu(x1*W_x) lives exactly on "
               "two rays, i.e. it only transmits a1 = relu(x1) and b1 = relu(-x1) (positive/negative part of the "
               "first input, at fixed ratios), so h1 is 2-dimensional (no other information about x1 survives); "
               "(iii) the later states grow one 2-dimensional ray per step (variance rank 2,~4,~6,8,8 at h1..h5) "
               "and are read out linearly (out = h5 W_o). Stages 2-5 are still the copied recurrence; the next "
               "step is to replace them by the rank/trace mechanism.")

px = np.array([0.643, 0.0, 0.423, 0.0, 0.045, 0.0, 0.0, 7.002])
nx = np.array([0.0, 6.421, 0.0, 6.574, 0.0, 4.913, 0.054, 0.0])
Wx = np.array([0.643,-6.421,0.423,-6.574,0.045,-4.913,-0.054,7.002])
Wh = np.array([
 [ 0.125, 0.155, 0.476, 0.061, 0.771, 0.036, 0.944,-0.163],
 [-0.465,-0.236, 0.272, 0.493, 0.169,-0.732, 0.424,-0.027],
 [ 0.020,-0.129,-0.107,-0.190,-0.618,-0.424, 0.240, 0.082],
 [ 0.528,-0.300,-0.363,-0.847, 0.279, 0.570, 0.199, 0.061],
 [-0.177,-0.324, 0.216,-0.252,-0.747,-0.192,-1.207, 0.380],
 [ 0.881, 0.600, 0.126, 0.348, 0.127, 0.070,-0.750, 0.024],
 [ 0.127, 0.962,-0.086, 0.973, 0.546, 0.680, 0.787,-0.890],
 [-1.656,-0.017, 1.419, 1.088, 0.156, 0.815, 1.102,-0.828]])
Wo = np.array([
 [ 0.716,-1.137, 0.680,-1.948, 0.727],
 [ 0.336,-0.716,-3.905, 3.885,-0.239],
 [ 0.849,-0.069,-3.387, 0.567, 0.633],
 [ 0.360,-2.426, 4.804,-2.857,-0.673],
 [-0.403, 2.160,-0.467,-0.468,-0.377],
 [-0.871, 4.200,-1.351,-1.310,-0.845],
 [-0.867,-0.938, 0.518, 0.679, 0.493],
 [ 0.240, 0.317, 0.378, 0.356,-2.253]])

def model(x, iv):
    a1 = iv("a1", np.maximum(x[:, 0], 0.0))
    b1 = iv("b1", np.maximum(-x[:, 0], 0.0))
    h = a1[:, None] * px[None, :] + b1[:, None] * nx[None, :]
    for t in range(1, 5):
        pre = x[:, t:t+1] * Wx[None, :] + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    return h @ Wo

ALIGN = {"a1": ["h1"], "b1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "h5": ["h5"]}
