import numpy as np
DESCRIPTION = ("Residual stream R = token embedding + position embedding. Layer 0: each position p forms a "
 "query R[p]M0 and key R[q] (M0 = W_Q0 W_K0^T); the bilinear score S[p,q] = R[p]M0.R[q]/2 is softmaxed over keys "
 "and the attended residual is added back through V0 = W_V0 W_O0. Layer 1 scores T[p,q] = (Z[p]M1).Z[q]/2 with "
 "M1 = W_Q1 W_K1^T and softmaxes over keys; only position 4 is read out: out = (Z[4] + a1.(Z V1O1)) @ W_U where "
 "a1 = row 4 of layer-1 attention and V1O1 = W_V1 W_O1. All matrices are the network's, as integer tables /1000.")
E = np.array([[-555,-2026,1424,301],[1022,-1087,678,1846],[999,-200,1144,1396],[1769,1896,1094,-522],[-640,-17,2882,-74]])/1000
P = np.array([[933,-525,2255,26],[1908,-1154,594,-1673],[-1298,1957,2546,-474],[-1592,-453,-67,2061],[-1571,-1151,170,2608]])/1000
M0 = np.array([[-16742,3424,-21421,-15363],[-20575,10823,-20548,8575],[851,5740,13216,10749],[11627,-7776,5781,-4136]])/1000
V0 = np.array([[1691,938,-494,453],[523,583,-140,760],[-601,1063,-147,-838],[459,1326,-258,1043]])/1000
M1 = np.array([[-291,-549,465,1231],[-166,-341,288,505],[-1556,-5278,6376,9313],[-259,-770,1061,1025]])/1000
V1 = np.array([[-533,11540,-3010,-161],[2118,-1621,-470,1404],[-168,16038,-1941,-3937],[-535,-7531,3190,-2708]])/1000
WU = np.array([[140,-102,-27,-30,-158],[-44,114,112,-139,-395],[689,-418,198,-527,-1465],[-315,-363,607,361,6]])/1000
def softmax(S):
    A = np.exp(S - S.max(-1, keepdims=True))
    return A / A.sum(-1, keepdims=True)
def model(x, iv):
    B = x.shape[0]
    e = iv('embed', (E[x] + P).reshape(B, -1)).reshape(B, 5, 4)
    A0 = softmax(np.einsum('bpd,bqd->bpq', e @ M0, e) / 2)
    A0 = iv('A0', A0.reshape(B, -1)).reshape(B, 5, 5)
    o0 = iv('o0', np.einsum('bpq,bqd,de->bpe', A0, e, V0).reshape(B, -1)).reshape(B, 5, 4)
    Z = iv('Z', (e + o0).reshape(B, -1)).reshape(B, 5, 4)
    A1 = softmax(np.einsum('bpd,bqd->bpq', Z @ M1, Z) / 2)
    a1 = iv('a1', A1[:, 4])
    o1 = iv('o1', np.einsum('bk,bkd,de->be', a1, Z, V1))
    r1 = iv('r1', Z[:, 4] + o1)
    return r1 @ WU
ALIGN = {'embed': ['embed'], 'A0': ['attn0.pattern'], 'o0': ['attn0.out'], 'Z': ['resid0.mid'], 'a1': ['attn1.pattern'], 'o1': ['attn1.out'], 'r1': ['resid1.mid']}
