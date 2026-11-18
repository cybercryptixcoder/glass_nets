
import numpy as np

DESCRIPTION = ("Diagnostic exact re-implementation (stage-by-stage compression follows). Residual x = W_E[tok]+W_pos. "
               "Layer l: logits = x W_Q W_K^T x^T / 2, A = softmax over keys, x += (A x) (W_V W_O). "
               "Only the products W_Q W_K^T and W_V W_O matter. out = x[4] @ W_U.")

WE = np.array([[-0.55,-2.03,1.42,0.30],[1.02,-1.09,0.68,1.85],[1.00,-0.20,1.14,1.40],
               [1.77,1.90,1.09,-0.52],[-0.64,-0.02,2.88,-0.07]])
WP = np.array([[0.93,-0.53,2.25,0.03],[1.91,-1.15,0.59,-1.67],[-1.30,1.96,2.55,-0.47],
               [-1.59,-0.45,-0.07,2.06],[-1.57,-1.15,0.17,2.61]])
WU = np.array([[0.14,-0.10,-0.03,-0.03,-0.16],[-0.04,0.11,0.11,-0.14,-0.40],
               [0.69,-0.42,0.20,-0.53,-1.46],[-0.32,-0.36,0.61,0.36,0.01]])
M0 = np.array([[-16.74,3.42,-21.42,-15.36],[-20.57,10.82,-20.55,8.58],
               [0.85,5.74,13.22,10.75],[11.63,-7.78,5.78,-4.14]])
M1 = np.array([[-0.29,-0.55,0.47,1.23],[-0.17,-0.34,0.29,0.51],
               [-1.56,-5.28,6.38,9.31],[-0.26,-0.77,1.06,1.02]])
V0 = np.array([[1.69,0.94,-0.49,0.45],[0.52,0.58,-0.14,0.76],[-0.60,1.06,-0.15,-0.84],[0.46,1.33,-0.26,1.04]])
V1 = np.array([[-0.53,11.54,-3.01,-0.16],[2.12,-1.62,-0.47,1.40],[-0.17,16.04,-1.94,-3.94],[-0.54,-7.53,3.19,-2.71]])

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
    x0 = np.stack(vs, 1)                                  # B,5,4
    A0 = sm(np.einsum('bik,kl,bjl->bij', x0, M0, x0) / 2.0)
    A0 = iv("A0", A0.reshape(B, 25))
    o0 = iv("o0", np.einsum('bij,bjk,kl->bil', A0.reshape(B,5,5), x0, V0).reshape(B, 20))
    x1 = iv("r0", x0 + o0.reshape(B, 5, 4))
    A1 = sm(np.einsum('bik,kl,bjl->bij', x1, M1, x1) / 2.0)
    A1 = iv("A1", A1.reshape(B, 25))
    o1 = iv("o1", np.einsum('bij,bjk,kl->bil', A1.reshape(B,5,5), x1, V1).reshape(B, 20))
    x2 = iv("r1", x1 + o1.reshape(B, 5, 4))
    return x2[:, 4, :] @ WU

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"], "e4": ["embed"],
         "A0": ["attn0.pattern"], "o0": ["attn0.out"], "r0": ["resid0.mid"],
         "A1": ["attn1.pattern"], "o1": ["attn1.out"], "r1": ["resid1.mid"]}
