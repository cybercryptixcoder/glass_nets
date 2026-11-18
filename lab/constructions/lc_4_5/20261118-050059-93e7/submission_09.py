import numpy as np

DESCRIPTION = ("Mechanism. Residual stream = W_E[token]+W_pos[pos] (embed). Layer 0: score(q,k)=r_q.M0.r_k/2, "
               "M0=W_Q W_K^T (rank 3, strongly non-symmetric); attention is nearly hard (mean max weight 0.87, "
               "86% of query rows >0.9): row q copies one key. Output = A0.R.V0, V0=W_V W_O (V0 maps the "
               "chosen key's residual). Layer 1: only the readout position 4 matters. Its query direction is "
               "qv=M1.rows, score(q=4,k)=qv.z_k/2, softmax over the 5 keys, and layer 1 adds "
               "sum_k A1_4k (z_k.V1) whose readout is sum_k A1_4k (z_k.G), G=V1 W_U. So logits = "
               "z4.W_U + softmax_k(qv.z_k/2) . (z_k.G): a soft max/min selector over the 5 key positions, "
               "weighted by a rank->2 class table G. Matrices are written as integers over 1000.")

WE = np.array([[-555,-2026,1424,301],[1022,-1087,678,1846],[999,-200,1144,1396],
               [1769,1896,1094,-522],[-640,-17,2882,-74]])/1000
WP = np.array([[933,-525,2255,26],[1908,-1154,594,-1673],[-1298,1957,2546,-474],
               [-1592,-453,-67,2061],[-1571,-1151,170,2608]])/1000
M0 = np.array([[-16742,3424,-21421,-15363],[-20575,10823,-20548,8575],
               [851,5740,13216,10749],[11627,-7776,5781,-4136]])/1000
V0 = np.array([[1691,938,-494,453],[523,583,-140,760],[-601,1063,-147,-838],
               [459,1326,-258,1043]])/1000
M1 = np.array([[-291,-549,465,1231],[-166,-341,288,505],[-1556,-5278,6376,9313],
               [-259,-770,1061,1025]])/1000
G  = np.array([[-2601,2690,614,-58,-72],[-400,-714,519,916,1003],
               [-820,4091,-971,-2618,-3495],[3306,-1157,-1841,-1597,-1626]])/1000
WU = np.array([[140,-102,-27,-30,-158],[-44,114,112,-139,-395],[689,-418,198,-527,-1465],
               [-315,-363,607,361,6]])/1000

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    R = np.zeros((B, 5, 4))
    for p in range(5):
        R[:, p, :] = iv("e%d" % p, WE[x[:, p]] + WP[p])
    S = R @ M0 @ R.transpose(0, 2, 1) / 2.0
    A0 = np.exp(S - S.max(axis=2, keepdims=True))
    A0 = iv("A0", (A0 / A0.sum(axis=2, keepdims=True)).reshape(B, 25)).reshape(B, 5, 5)
    o0 = iv("o0", (A0 @ R @ V0).reshape(B, 20)).reshape(B, 5, 4)
    Z = iv("Z", (R + o0).reshape(B, 20)).reshape(B, 5, 4)
    qv = Z @ M1
    T = (qv[:, :, None, :] * Z[:, None, :, :]).sum(-1) / 2.0
    A1 = np.exp(T - T.max(axis=2, keepdims=True))
    A1 = A1 / A1.sum(axis=2, keepdims=True)
    a1 = iv("a1", A1[:, 4, :])
    return Z[:, 4, :] @ WU + (a1[:, :, None] * (Z @ G)).sum(1)

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"], "e4": ["embed"],
         "A0": ["attn0.pattern"], "o0": ["attn0.out"], "Z": ["resid0.mid"],
         "a1": ["attn1.pattern"]}
