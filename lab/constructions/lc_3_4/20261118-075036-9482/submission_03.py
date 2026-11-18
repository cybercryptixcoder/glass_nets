
import numpy as np
DESCRIPTION = ("embed = token+position embeddings. Layer0: one softmax head with logits e_i^T A0 e_j/sqrt3; its "
               "output adds a softmax-mixed value to the residual (resid0 site). Layer1: one softmax head with "
               "logits h_i^T A1 h_j/sqrt3; the last position reads out resid0_last + sum_j a1_j (h_j M) @ W_U.")
WE = np.array([[1.303,0.258,0.942],[0.828,0.076,1.370],[-1.594,1.030,0.315],[0.171,0.428,-0.962]])
WP = np.array([[1.415,1.799,2.992],[-0.132,-0.655,1.292],[-4.175,-0.700,-0.066],[-0.105,1.203,0.867]])
A0 = np.array([[-0.928,1.067,-0.941],[1.714,5.098,-2.799],[1.273,-0.173,0.058]])
VO0 = np.array([[-0.300,1.429,0.166],[0.055,-0.907,-0.371],[0.011,-0.335,-0.010]])
A1 = np.array([[-0.216,-0.351,17.637],[1.032,0.317,0.002],[-1.326,-0.879,-1.196]])
M = np.array([[-4.576,1.731,3.553],[-6.691,3.445,4.541],[2.659,-0.151,-4.114]])
WU = np.array([[2.915,-3.388,1.144,0.247],[11.061,12.343,-3.341,1.325],[-1.261,2.389,-24.543,12.903]])

def _sm(S):
    A = np.exp(S - S.max(-1, keepdims=True)); return A / A.sum(-1, keepdims=True)

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    e = iv("emb", (WE[x] + WP).reshape(B, 12)).reshape(B, 4, 3)
    a0 = _sm(np.einsum('bid,de,bje->bij', e, A0, e) / np.sqrt(3.0))
    h0 = e + np.einsum('bij,bjd,de->bie', a0, e, VO0)
    h0 = iv("r0", h0.reshape(B, 12)).reshape(B, 4, 3)
    p1 = _sm(np.einsum('bid,de,bje->bij', h0, A1, h0) / np.sqrt(3.0))
    p1 = iv("p1", p1[:, :, :3].reshape(B, 12)).reshape(B, 4, 3)
    p1 = np.concatenate([p1, 1.0 - p1.sum(-1, keepdims=True)], axis=-1)
    h1 = h0 + np.einsum('bij,bjd,de->bie', p1, h0, M)
    return h1[:, -1] @ WU

ALIGN = {"emb": ["embed"], "r0": ["resid0.mid"], "p1": ["attn1.pattern"]}
