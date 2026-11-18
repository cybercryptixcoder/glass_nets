
import numpy as np

DESCRIPTION = ("Stage embedding: at each position the residual stream is that position's token embedding plus "
               "the fixed position embedding (one 5-dim one-hot per position). Stage layer-0 attention: the "
               "head's score for query (i, ti) and key (j, tj) is a sum of four small tables measured from the "
               "network - token-pair tt[ti,tj], query-token/key-position tp[ti,j], query-position/key-token "
               "pt[i,tj] and position-pair pp[i,j] - and the pattern is its softmax over keys. Readout not yet "
               "modelled (crude W_U on the last position's embedding).")

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

TT = np.array([[ 0.02, -0.05, -0.37, -0.21,  0.06],
               [ 0.49, -0.21, -1.63, -0.63, -0.55],
               [ 1.73, -0.31, -2.94, -1.55, -1.14],
               [ 1.90, -0.30, -3.53, -1.92, -1.19],
               [-1.70,  0.28,  3.68,  1.96,  1.10]])
PP = np.array([[-1.94, -4.18,  0.48,  9.24, -1.85],
               [-0.68,  2.22,  0.73, -1.69, -0.04],
               [-3.37,  0.04, -2.26,  6.18, -2.22],
               [ 3.36,  1.29,  0.30, -9.17,  3.53],
               [-0.46,  1.00,  0.27, -0.35, -0.66]])
TP = np.array([[ 0.17,  0.19, -0.35, -0.10, -0.21],
               [-1.21,  1.36, -0.22,  1.53, -1.72],
               [-2.39,  0.28, -2.48,  6.01, -3.48],
               [-2.62,  0.60, -2.74,  6.60, -3.92],
               [ 2.47, -1.19,  2.37, -5.81,  3.85]])
PT = np.array([[ 2.14,  0.21,  0.77, -0.18, -1.41],
               [-0.43,  0.06, -0.73, -0.09,  0.01],
               [ 1.60,  0.22, -1.93, -1.09, -1.16],
               [-2.28,  0.03,  1.67,  1.08,  1.73],
               [-0.02, -0.23, -0.58,  0.00, -0.24]])

def model(x, iv):
    t = x.astype(int)
    toks = [iv("tok%d" % p, np.eye(5)[t[:, p]]) for p in range(5)]
    sc = (TT[t[:, :, None], t[:, None, :]] + TP[t] + PT[:, t] + PP[None, :, :])
    sc = sc - sc.max(2, keepdims=True)
    ex = np.exp(sc)
    pat0 = iv("pat0", (ex / ex.sum(2, keepdims=True)).reshape(-1, 25))
    e4 = iv("emb_last", toks[4] @ W_E + W_pos[4])
    return e4 @ W_U

ALIGN = {"tok0": ["embed"], "tok1": ["embed"], "tok2": ["embed"],
         "tok3": ["embed"], "tok4": ["embed"], "pat0": ["attn0.pattern"],
         "emb_last": ["resid0.mid"]}
