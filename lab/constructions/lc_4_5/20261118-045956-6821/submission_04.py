
import numpy as np

DESCRIPTION = ("Two attention steps over the residual stream, written with the QK and OV products only (the residual basis is "
  "a gauge freedom). embed = token+position embedding. Layer-0 attention scores query/key embeddings with the 4x4 QK0 "
  "(logits of order 100, so its softmax is essentially a hard pointer), and writes OV0 applied to the average of the "
  "attended embeddings. Layer-1 attention scores the residuals with QK1 (nearly rank 1) and writes OV1 applied to the "
  "average of the attended residuals. The last position's residual is unembedded to 5 class scores.")

WE = np.array([[-0.555,-2.026,1.424,0.301],[1.022,-1.087,0.678,1.846],
               [0.999,-0.2,1.144,1.396],[1.769,1.896,1.094,-0.522],
               [-0.64,-0.017,2.882,-0.074]])
WP = np.array([[0.933,-0.525,2.255,0.026],[1.908,-1.154,0.594,-1.673],
               [-1.298,1.957,2.546,-0.474],[-1.592,-0.453,-0.067,2.061],
               [-1.571,-1.151,0.17,2.608]])
WU = np.array([[0.14,-0.102,-0.027,-0.03,-0.158],[-0.044,0.114,0.112,-0.139,-0.395],
               [0.689,-0.418,0.198,-0.527,-1.465],[-0.315,-0.363,0.607,0.361,0.006]])
QK0 = np.array([[-16.742,3.424,-21.421,-15.363],[-20.575,10.823,-20.548,8.575],
                [0.851,5.74,13.216,10.749],[11.627,-7.776,5.781,-4.136]])
QK1 = np.array([[-0.291,-0.549,0.465,1.231],[-0.166,-0.341,0.288,0.505],
                [-1.556,-5.278,6.376,9.313],[-0.259,-0.77,1.061,1.025]])
OV0 = np.array([[1.691,0.938,-0.494,0.453],[0.523,0.583,-0.14,0.76],
                [-0.601,1.063,-0.147,-0.838],[0.459,1.326,-0.258,1.043]])
OV1 = np.array([[-0.533,11.54,-3.01,-0.161],[2.118,-1.621,-0.47,1.404],
                [-0.168,16.038,-1.941,-3.937],[-0.535,-7.531,3.19,-2.708]])

def _softmax(L):
    L = L - L.max(axis=2, keepdims=True)
    A = np.exp(L)
    return A / A.sum(axis=2, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    emb = WP[None] + WE[x]
    e = iv("emb", emb.reshape(B, 20))
    emb = e.reshape(B, 5, 4)
    A0 = _softmax((emb @ QK0) @ emb.transpose(0, 2, 1) / 2.0)
    p0 = iv("pat0", A0.reshape(B, 25))
    a0 = iv("aout0", (p0.reshape(B, 5, 5) @ emb @ OV0).reshape(B, 20))
    h = iv("res0", e + a0).reshape(B, 5, 4)
    A1 = _softmax((h @ QK1) @ h.transpose(0, 2, 1) / 2.0)
    p1 = iv("pat1", A1.reshape(B, 25))
    a1 = iv("aout1", (p1.reshape(B, 5, 5) @ h @ OV1).reshape(B, 20))
    r1 = iv("res1", (e + a0) + a1)
    return r1.reshape(B, 5, 4)[:, -1, :] @ WU

ALIGN = {"emb": ["embed"], "pat0": ["attn0.pattern"], "aout0": ["attn0.out"],
         "res0": ["resid0.mid"], "pat1": ["attn1.pattern"], "aout1": ["attn1.out"],
         "res1": ["resid1.mid"]}
