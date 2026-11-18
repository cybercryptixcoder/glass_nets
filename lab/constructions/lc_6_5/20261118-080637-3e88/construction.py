
import numpy as np

DESCRIPTION = ("Diagnostic first pass. Stage 'embedding': at each position the residual stream is the "
               "token embedding row (one-hot over the 5 tokens) plus the fixed position embedding, so "
               "each position is described by its 5-dim token one-hot and a constant offset. The readout "
               "uses W_U applied to the last position's embedding (attention contributions not yet modelled).")

W_E = np.array([
 [-0.2981, -0.2268, -0.0179, 0.0432, -0.4107, -0.1951],
 [-0.1166,  0.2751,  0.0365, -0.8035, -0.0855,  0.5458],
 [-0.6288, -0.7465,  1.4146, -0.2421,  0.2873,  1.8283],
 [-0.6147, -1.1474,  0.4567, -0.0289, -0.0583,  1.6885],
 [ 0.6221,  1.1213,  0.3178,  0.0459,  0.7242, -1.2436]])

W_pos = np.array([
 [ 2.0293, -1.2658, -0.0211,  0.8249,  0.0348,  1.3073],
 [ 0.1668,  0.6632, -1.8315, -0.28,   -0.1254, -0.1004],
 [-0.2908, -0.705,   0.0844,  1.666,  -0.5198,  2.4516],
 [-1.609,   0.7625,  0.761,   0.1783, -1.0029, -1.9958],
 [-0.065,   0.8791,  1.0643, -1.2339,  0.9434,  0.4781]])

W_U = np.array([
 [ 3.3589,  3.626,  -0.1343,  3.7195, -4.1399],
 [ 2.0978, -3.4111, -2.4337,  2.2538, -7.5581],
 [ 0.8778,  4.5569, -3.78,    1.6338, -3.5775],
 [-4.7942, -2.8417, -0.1682,  5.2252,  0.0318],
 [ 2.2955,  0.2095,  5.962,  -2.8674, -6.3147],
 [ 1.1823,  0.4164,  2.3148,  4.3166, -8.9681]])

def model(x, iv):
    t = x.astype(int)
    toks = []
    for p in range(5):
        toks.append(iv("tok%d" % p, np.eye(5)[t[:, p]]))
    e4 = iv("emb_last", toks[4] @ W_E + W_pos[4])
    return e4 @ W_U

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "emb_last": ["resid0.mid"]}
