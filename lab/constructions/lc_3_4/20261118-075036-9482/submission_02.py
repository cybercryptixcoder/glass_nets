
import numpy as np
DESCRIPTION = ("embedding stage: token+position embeddings per position (12 numbers). layer 0: one softmax "
               "attention head whose logits are the bilinear form e_i^T A0 e_j / sqrt(3); the pattern is the "
               "softmax over keys. Layer 1 (currently copied) is a second head over resid0 feeding W_U.")
WE = np.array([[1.303,0.258,0.942],[0.828,0.076,1.370],[-1.594,1.030,0.315],[0.171,0.428,-0.962]])
WP = np.array([[1.415,1.799,2.992],[-0.132,-0.655,1.292],[-4.175,-0.700,-0.066],[-0.105,1.203,0.867]])
A0 = np.array([[-0.928,1.067,-0.941],[1.714,5.098,-2.799],[1.273,-0.173,0.058]])
VO0 = np.array([[-0.300,1.429,0.166],[0.055,-0.907,-0.371],[0.011,-0.335,-0.010]])
A1 = np.array([[-0.216,-0.351,17.637],[1.032,0.317,0.002],[-1.326,-0.879,-1.196]])
W1V = np.array([[2.046,0.050,-0.276],[2.948,-0.677,0.196],[-1.375,-1.169,-1.116]])
W1O = np.array([[-2.226,0.900,1.762],[0.226,-1.115,1.079],[0.123,0.194,0.385]])
WU = np.array([[2.915,-3.388,1.144,0.247],[11.061,12.343,-3.341,1.325],[-1.261,2.389,-24.543,12.903]])

def _sm(S):
    A = np.exp(S - S.max(-1, keepdims=True)); return A / A.sum(-1, keepdims=True)

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    e = iv("emb", (WE[x] + WP).reshape(B, 12)).reshape(B, 4, 3)
    S0 = np.einsum('bid,de,bje->bij', e, A0, e) / np.sqrt(3.0)
    a0 = _sm(S0)
    a0 = iv("att0", a0[:, :, :3].reshape(B, 12)).reshape(B, 4, 3)
    a0 = np.concatenate([a0, 1.0 - a0.sum(-1, keepdims=True)], axis=-1)
    h0 = e + np.einsum('bij,bjd,de->bie', a0, e, VO0)
    q = h0 @ (A1 @ np.linalg.inv(np.eye(3)) * 0)  # placeholder unused
    S1 = np.einsum('bid,de,bje->bij', h0, A1, h0) / np.sqrt(3.0)
    a1 = _sm(S1)
    h1 = h0 + np.einsum('bij,bjd,de->bie', a1, h0 @ W1V, W1O)
    return h1[:, -1] @ WU

ALIGN = {"emb": ["embed"], "att0": ["attn0.pattern"]}
