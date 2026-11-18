import numpy as np
DESCRIPTION = "Exact-arithmetic account of the 2-layer attention-only transformer, written in integer tables (x/1000). Residual R = WE[token]+WP. Layer0: score = R M0 R^T, softmax over keys, adds A0 R V0. Layer1: score = (Z M1).Z^T, softmax over keys, adds A1 (Z V1) at every position. Output = resid1[last] @ WU. All matrices are the reduced products (M0=QK^T/2, etc.)."
WEi = np.array([[-555,-2026,1424,301],[1022,-1087,678,1846],[999,-200,1144,1396],[1769,1896,1094,-522],[-640,-17,2882,-74]])
WPi = np.array([[933,-525,2255,26],[1908,-1154,594,-1673],[-1298,1957,2546,-474],[-1592,-453,-67,2061],[-1571,-1151,170,2608]])
M0i = np.array([[-8371,1712,-10711,-7681],[-10287,5411,-10274,4288],[425,2870,6608,5375],[5813,-3888,2890,-2068]])
V0i = np.array([[1691,938,-494,453],[523,583,-140,760],[-601,1063,-147,-838],[459,1326,-258,1043]])
M1i = np.array([[-146,-274,233,615],[-83,-171,144,253],[-778,-2639,3188,4657],[-129,-385,531,512]])
V1i = np.array([[-533,11540,-3010,-161],[2118,-1621,-470,1404],[-168,16038,-1941,-3937],[-535,-7531,3190,-2708]])
WUi = np.array([[140,-102,-27,-30,-158],[-44,114,112,-139,-395],[689,-418,198,-527,-1465],[-315,-363,607,361,6]])

def model(x, iv):
    B = x.shape[0]
    R = iv('embed', (WEi[x] + WPi).reshape(B, -1)).reshape(B, 5, 4) / 1000.
    S = R @ M0i @ R.transpose(0, 2, 1) / 1000.
    S = S - S.max(-1, keepdims=True); A0 = np.exp(S); A0 = A0 / A0.sum(-1, keepdims=True)
    A0 = iv('pat0', A0.reshape(B, -1)).reshape(B, 5, 5)
    o0 = iv('o0', (A0 @ R @ V0i / 1000.).reshape(B, -1)).reshape(B, 5, 4)
    Z = iv('z', (R + o0).reshape(B, -1)).reshape(B, 5, 4)
    T = Z @ M1i / 1000.
    T = (T[:, :, None, :] * Z[:, None, :, :]).sum(-1)
    T = T - T.max(-1, keepdims=True); A1 = np.exp(T); A1 = A1 / A1.sum(-1, keepdims=True)
    A1 = iv('pat1', A1.reshape(B, -1)).reshape(B, 5, 5)
    o1 = iv('o1', (A1 @ (Z @ V1i / 1000.)).reshape(B, -1)).reshape(B, 5, 4)
    r1 = iv('r1', (Z + o1).reshape(B, -1)).reshape(B, 5, 4)
    return r1[:, 4] @ WUi / 1000.

ALIGN = {'embed': ['embed'], 'pat0': ['attn0.pattern'], 'o0': ['attn0.out'], 'z': ['resid0.mid'], 'pat1': ['attn1.pattern'], 'o1': ['attn1.out'], 'r1': ['resid1.mid']}
