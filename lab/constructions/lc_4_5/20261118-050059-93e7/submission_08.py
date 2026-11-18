import numpy as np

DESCRIPTION = ("Full mechanism, structurally reduced. Residual = W_E[token]+W_pos[pos]. Layer 0: bilinear "
               "score r_q^T M0 r_k /2 (M0 = QK product), softmax over keys -> attention pattern A0, output "
               "A0 R V0 (V0 = OV product). Layer 1: same shape with M1, softmax -> A1; only the readout "
               "position 4 matters, and its output goes through G = V1 W_U, so logits = resid1[4] @ W_U "
               "= z4 @ W_U + sum_k A1_4k (z_k @ G). Layer-0 attention is nearly hard (one key wins).")

WE = np.array([[-0.5547,-2.0260, 1.4238, 0.3010],
               [ 1.0222,-1.0874, 0.6785, 1.8459],
               [ 0.9991,-0.2001, 1.1441, 1.3961],
               [ 1.7693, 1.8963, 1.0939,-0.5216],
               [-0.6405,-0.0166, 2.8816,-0.0738]])
WP = np.array([[ 0.9330,-0.5251, 2.2547, 0.0259],
               [ 1.9077,-1.1536, 0.5939,-1.6735],
               [-1.2981, 1.9569, 2.5465,-0.4738],
               [-1.5923,-0.4525,-0.0671, 2.0607],
               [-1.5712,-1.1513, 0.1696, 2.6083]])
M0 = np.array([[-16.742, 3.4242,-21.421,-15.363],
               [-20.575,10.8229,-20.548, 8.5752],
               [  0.8506,5.7401,13.216,10.749],
               [ 11.627,-7.7763, 5.7809,-4.1356]])
V0 = np.array([[ 1.6914, 0.9376,-0.4944, 0.4528],
               [ 0.5231, 0.5833,-0.1400, 0.7603],
               [-0.6005, 1.0629,-0.1470,-0.8383],
               [ 0.4595, 1.3260,-0.2580, 1.0431]])
M1 = np.array([[-0.2913,-0.5490, 0.4652, 1.2309],
               [-0.1663,-0.3412, 0.2876, 0.5054],
               [-1.5564,-5.2785, 6.3757, 9.3132],
               [-0.2590,-0.7697, 1.0613, 1.0249]])
G  = np.array([[-2.6012, 2.6897, 0.6144,-0.0577,-0.0716],
               [-0.3996,-0.7141, 0.5193, 0.9165, 1.0031],
               [-0.8197, 4.0906,-0.9707,-2.6184,-3.4953],
               [ 3.3055,-1.1569,-1.8407,-1.5973,-1.6260]])
WU = np.array([[ 0.1397,-0.1018,-0.0272,-0.0296,-0.1579],
               [-0.0436, 0.1143, 0.1122,-0.1387,-0.3954],
               [ 0.6891,-0.4180, 0.1983,-0.5265,-1.4645],
               [-0.3152,-0.3630, 0.6068, 0.3611, 0.0060]])

ROWS = np.array([3, 4])

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
    Z4 = Z[:, 4, :]
    T = (Z @ M1[ROWS].T).transpose(0, 2, 1) * Z[:, None, :, :]
    T = T.sum(-1) / 2.0
    A1p = np.exp(T - T.max(axis=2, keepdims=True))
    A1p = iv("A1p", (A1p / A1p.sum(axis=2, keepdims=True)).reshape(B, 10)).reshape(B, 2, 5)
    a1 = A1p[:, 1, :]
    o1 = iv("o1", (a1[:, :, None] * (Z @ G)).sum(1))
    r1 = iv("r1", Z4 + o1)
    return r1 @ WU

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"], "e4": ["embed"],
         "A0": ["attn0.pattern"], "o0": ["attn0.out"], "Z": ["resid0.mid"],
         "A1p": ["attn1.pattern"], "o1": ["attn1.out"], "r1": ["resid1.mid"]}
