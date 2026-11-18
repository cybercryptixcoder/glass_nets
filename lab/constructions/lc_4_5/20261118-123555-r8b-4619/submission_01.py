import numpy as np
DESCRIPTION = ("Integer-table copy (scaled 1/1000) of the reduced exact computation: layer-0 attention "
 "scores from a bilinear form of the residual, softmax, add attended values, layer-1 attention from a "
 "rank-rich bilinear form, add, read out position 4 through W_U. No rules yet.")
WEI = np.array([[-555,-2026,1424,301],
 [1022,-1087,678,1846],
 [999,-200,1144,1396],
 [1769,1896,1094,-522],
 [-640,-17,2882,-74]])
WPI = np.array([[933,-525,2255,26],
 [1908,-1154,594,-1673],
 [-1298,1957,2546,-474],
 [-1592,-453,-67,2061],
 [-1571,-1151,170,2608]])
M0I = np.array([[-16742,3424,-21421,-15363],
 [-20575,10823,-20548,8575],
 [851,5740,13216,10749],
 [11627,-7776,5781,-4136]])
V0I = np.array([[1691,938,-494,453],
 [523,583,-140,760],
 [-601,1063,-147,-838],
 [459,1326,-258,1043]])
M1I = np.array([[-291,-549,465,1231],
 [-166,-341,288,505],
 [-1556,-5278,6376,9313],
 [-259,-770,1061,1025]])
V1I = np.array([[-533,11540,-3010,-161],
 [2118,-1621,-470,1404],
 [-168,16038,-1941,-3937],
 [-535,-7531,3190,-2708]])
WUI = np.array([[140,-102,-27,-30,-158],
 [-44,114,112,-139,-395],
 [689,-418,198,-527,-1465],
 [-315,-363,607,361,6]])

def model(x, iv):
    B = x.shape[0]
    R = iv('embed', ((WEI[x] + WPI[None])/1000).reshape(B,20)).reshape(B,5,4)
    S0 = np.einsum('bpd,de,bqe->bpq', R, M0I, R)/2000
    A0 = np.exp(S0 - S0.max(-1,keepdims=True)); A0 = A0/A0.sum(-1,keepdims=True)
    A0 = iv('pattern0', A0.reshape(B,25)).reshape(B,5,5)
    o0 = np.einsum('bpq,bqe,ed->bpd', A0, R, V0I)/1000
    o0 = iv('attn_out0', o0.reshape(B,20)).reshape(B,5,4)
    Z = iv('resid_mid0', (R+o0).reshape(B,20)).reshape(B,5,4)
    qv = np.einsum('bpd,de->bpe', Z, M1I)/1000
    T = np.einsum('bpd,bqd->bpq', qv, Z)/2
    A1 = np.exp(T - T.max(-1,keepdims=True)); A1 = A1/A1.sum(-1,keepdims=True)
    A1 = iv('pattern1', A1.reshape(B,25)).reshape(B,5,5)
    o1 = np.einsum('bpq,bqe,ed->bpd', A1, Z, V1I)/1000
    o1 = iv('attn_out1', o1.reshape(B,20)).reshape(B,5,4)
    r1 = iv('resid_mid1', (Z+o1).reshape(B,20)).reshape(B,5,4)
    return r1[:,4] @ WUI / 1000

ALIGN = {'embed': ['embed'], 'pattern0': ['attn0.pattern'], 'attn_out0': ['attn0.out'],
         'resid_mid0': ['resid0.mid'], 'pattern1': ['attn1.pattern'], 'attn_out1': ['attn1.out'],
         'resid_mid1': ['resid1.mid']}
