
import numpy as np

DESCRIPTION = ("Exact re-implementation (all stages verified). Residual x = W_E[tok]+W_pos. Layer l: logits = "
               "x W_Q W_K^T x^T/2, A = softmax over keys, x += (A x)(W_V W_O). out = x[4] @ W_U. "
               "Findings so far: layer-0 attention is nearly one-hot (hard selection of one key per query); "
               "layer-1 QK is nearly rank 1 (12.8 u v^T, next 0.44); the readout V1 W_U is nearly rank 2 and its "
               "two directions are ~ u and ~ v, i.e. layer 1 and the readout share the same two scalar projections. "
               "Compression still pending: every coordinate of the 20-dim embedding affects the decision.")

WE = np.array([[-0.5547,-2.0260,1.4238,0.3010],[1.0222,-1.0874,0.6785,1.8459],[0.9991,-0.2001,1.1441,1.3961],
               [1.7693,1.8963,1.0939,-0.5216],[-0.6405,-0.0166,2.8816,-0.0738]])
WP = np.array([[0.9330,-0.5251,2.2547,0.0259],[1.9077,-1.1536,0.5939,-1.6735],[-1.2981,1.9569,2.5465,-0.4738],
               [-1.5923,-0.4525,-0.0671,2.0607],[-1.5712,-1.1513,0.1696,2.6083]])
WU = np.array([[0.1397,-0.1018,-0.0272,-0.0296,-0.1579],[-0.0436,0.1143,0.1122,-0.1387,-0.3954],
               [0.6891,-0.4180,0.1983,-0.5265,-1.4645],[-0.3152,-0.3630,0.6068,0.3611,0.0060]])
M0 = np.array([[-16.7420,3.4242,-21.4213,-15.3627],[-20.5746,10.8229,-20.5484,8.5752],
               [0.8506,5.7401,13.2163,10.7495],[11.6265,-7.7763,5.7809,-4.1356]])
M1 = np.array([[-0.2913,-0.5490,0.4652,1.2309],[-0.1663,-0.3412,0.2876,0.5054],
               [-1.5564,-5.2785,6.3757,9.3132],[-0.2590,-0.7697,1.0613,1.0249]])
V0 = np.array([[1.6914,0.9376,-0.4944,0.4528],[0.5231,0.5833,-0.1400,0.7603],
               [-0.6005,1.0629,-0.1470,-0.8383],[0.4595,1.3260,-0.2580,1.0431]])
V1 = np.array([[-0.5335,11.5397,-3.0097,-0.1606],[2.1180,-1.6214,-0.4697,1.4038],
               [-0.1676,16.0379,-1.9413,-3.9371],[-0.5355,-7.5311,3.1901,-2.7079]])

def sm(S):
    S = S - S.max(-1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(-1, keepdims=True)

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    vs = []
    for p in range(5):
        oh = np.zeros((B, 5)); oh[np.arange(B), x[:, p]] = 1.0
        vs.append(iv("e%d" % p, oh @ WE + WP[p]))
    x0 = np.stack(vs, 1)
    A0 = sm(np.einsum('bik,kl,bjl->bij', x0, M0, x0) / 2.0)
    A0 = iv("A0", A0.reshape(B, 25))
    o0 = iv("o0", np.einsum('bij,bjk,kl->bil', A0.reshape(B,5,5), x0, V0).reshape(B, 20))
    x1 = iv("r0", (x0 + o0.reshape(B, 5, 4)).reshape(B, 20)).reshape(B, 5, 4)
    A1 = sm(np.einsum('bik,kl,bjl->bij', x1, M1, x1) / 2.0)
    A1 = iv("A1", A1.reshape(B, 25))
    o1 = iv("o1", np.einsum('bij,bjk,kl->bil', A1.reshape(B,5,5), x1, V1).reshape(B, 20))
    x2 = iv("r1", (x1 + o1.reshape(B, 5, 4)).reshape(B, 20)).reshape(B, 5, 4)
    return x2[:, 4, :] @ WU

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"], "e4": ["embed"],
         "A0": ["attn0.pattern"], "o0": ["attn0.out"], "r0": ["resid0.mid"],
         "A1": ["attn1.pattern"], "o1": ["attn1.out"], "r1": ["resid1.mid"]}
