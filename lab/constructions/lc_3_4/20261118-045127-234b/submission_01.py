
import numpy as np
DESCRIPTION = "Diagnostic exact re-implementation of the 2-layer attention-only transformer. Stage-by-stage, variables are the site activations themselves; later stages are computed from earlier variables through the same attention arithmetic."

W_E = np.array([[-1.594,-0.9618,0.2582],[1.0295,0.8282,0.0764],[0.3151,0.9424,0.4281],[1.3033,0.1706,1.3696]])
W_pos = np.array([[0.8674,-0.7002,-0.6549],[-0.0662,-0.1318,2.992],[-4.1745,1.2921,1.2031],[1.7989,1.4152,-0.1054]])
W_U = np.array([[12.9028,-24.5428,11.0611,12.3428],[2.915,-3.3877,0.2467,1.1436],[2.3887,1.3247,-3.3413,-1.2613]])
Q0=np.array([[0.51,0.16,0.17],[1.36,-0.85,-0.24],[0.63,-2.85,0.99]])
K0=np.array([[0.33,1.59,-0.57],[-1.36,0.54,-0.48],[0.88,-1.48,-0.2]])
V0=np.array([[0.35,0.11,0.56],[1.18,-0.32,-0.08],[0.28,0.04,-0.97]])
O0=np.array([[0.88,-0.21,0.17],[0.05,-0.13,-0.17],[-0.51,-1.22,0.15]])
Q1=np.array([[-0.32,-0.98,0.0],[0.25,0.75,-0.0],[-3.68,0.47,0.0]])
K1=np.array([[0.07,-0.48,-0.0],[-1.22,-4.79,-0.0],[-1.78,0.1,-0.0]])
V1=np.array([[-1.17,-0.68,-1.38],[0.2,-0.28,0.05],[2.05,2.95,-1.12]])
O1=np.array([[0.23,-1.11,0.39],[0.19,-2.23,1.76],[1.08,0.9,0.12]])

def _sm(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    emb = W_E[x] + W_pos[None, :, :]
    e = iv("embed", emb.reshape(B, 12))
    h = e.reshape(B, 4, 3)
    q = h @ Q0; k = h @ K0; v = h @ V0
    p0 = _sm(q @ k.transpose(0, 2, 1) / np.sqrt(3.0))
    p0 = iv("p0", p0.reshape(B, 16)).reshape(B, 4, 4)
    ao = iv("ao", ((p0 @ v) @ O0).reshape(B, 12)).reshape(B, 4, 3)
    h = h + ao
    r0 = iv("r0", h.reshape(B, 12)).reshape(B, 4, 3)
    q = h @ Q1; k = h @ K1; v = h @ V1
    p1 = _sm(q @ k.transpose(0, 2, 1) / np.sqrt(3.0))
    p1 = iv("p1", p1.reshape(B, 16)).reshape(B, 4, 4)
    a1 = iv("a1", ((p1 @ v) @ O1).reshape(B, 12)).reshape(B, 4, 3)
    h = h + a1
    r1 = iv("r1", h.reshape(B, 12)).reshape(B, 4, 3)
    return r1[:, 3, :] @ W_U

ALIGN = {"embed": ["embed"], "p0": ["attn0.pattern"], "ao": ["attn0.out"],
         "r0": ["resid0.mid"], "p1": ["attn1.pattern"], "a1": ["attn1.out"],
         "r1": ["resid1.mid"]}
