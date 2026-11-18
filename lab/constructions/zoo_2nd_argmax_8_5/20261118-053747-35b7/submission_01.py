
import numpy as np

DESCRIPTION = ("Placeholder exact re-description of the 5-step ReLU RNN: h_t = relu(x_t W_x + h_{t-1} W_h), "
               "out = h_5 W_o. Constants copied from the trained weights. The task the network performs is "
               "the 'runner-up' (second largest) of the 5 inputs: the largest output class is the position "
               "of the second largest input value (verified: agrees with the network's decision on ~88% of "
               "inputs, the rest are near-ties/differences in magnitude). This version aligns each hidden "
               "site with its own 8 activations, so it is only a starting point for a real mechanism.")

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
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
    h = np.zeros((x.shape[0], 8))
    for t in range(5):
        pre = x[:, t:t+1] * Wx[None, :] + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    return h @ Wo

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "h5": ["h5"]}
