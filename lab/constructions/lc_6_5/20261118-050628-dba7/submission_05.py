
import numpy as np

DESCRIPTION = ("Partial. embed = W_E[token]+W_pos[position], exactly blockwise-linear in the five one-hot token "
               "indicators (one block per position); named at the embed site. Readout still crude (last-position "
               "embedding through W_U). Attention stages pending: I now have an exact forward pass in my scratch "
               "session and am measuring the layer-0/layer-1 low-rank maps to name their variables next.")

W_E = np.array([
 [-0.242, 0.724, -0.298, -0.629, -0.195, -0.029],
 [-0.803, 0.457,  1.415,  1.121, -0.411, -0.227],
 [ 1.828, -1.147, 1.689, -1.244, 0.275, -0.058],
 [-0.085, 0.287, -0.615, 0.043, 0.622, 0.036],
 [ 0.046, 0.546, -0.018, 0.318, -0.117, -0.746]])

W_pos = np.array([
 [ 0.762, -0.52,  -0.705,  1.666, 0.167,  0.879],
 [ 0.761,  2.029,  1.307, -0.28,  0.825, -1.266],
 [-1.832, -1.003,  0.663, -1.609, 1.064, -1.234],
 [-0.125, -0.065,  0.084, -0.021, -1.996, -0.291],
 [ 0.178,  0.035,  0.943,  0.478, -0.1,    2.452]])

W_U = np.array([
 [-6.315,  3.72,   5.962, -3.411, -2.867],
 [-4.14,   5.225,  2.098, -3.577, -2.842],
 [-3.78,   4.557,  0.21,  -0.134,  0.032],
 [-8.968,  0.878,  4.317,  2.295,  1.634],
 [ 3.626, -4.794, -2.434,  2.315,  2.254],
 [-7.558, -0.168,  3.359,  1.182,  0.416]])

def model(x, iv):
    B = x.shape[0]
    toks = []
    for p in range(5):
        oh = np.zeros((B, 5))
        oh[np.arange(B), x[:, p].astype(int)] = 1.0
        toks.append(iv("tok%d" % p, oh))
    last = toks[4] @ W_E + W_pos[4]
    logits = last @ W_U
    logits[:, 4] *= 0.2
    return logits

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"]}
