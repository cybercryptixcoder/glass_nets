
import numpy as np

DESCRIPTION = ("Compressed diagnostic: attention is described by its QK products only, and the OV term as "
  "'average the key residuals, then apply OV'. Layer-0 QK is numerically rank 3, layer-1 QK is numerically rank 1, "
  "so both are used in truncated SVD form (6 and 2 vectors). Note layer-0 attention logits are of order 100, so its "
  "softmax is a hard argmax: layer 0 acts as a pointer. Layer 1 scores keys by one scalar (projection of key) times "
  "one scalar (projection of query). The last position's residual is unembedded.")

WE = np.array([[-0.5547,-2.026,1.4238,0.301],[1.0222,-1.0874,0.6785,1.8459],
               [0.9991,-0.2001,1.1441,1.3961],[1.7693,1.8963,1.0939,-0.5216],
               [-0.6405,-0.0166,2.8816,-0.0738]])
WP = np.array([[0.933,-0.5251,2.2547,0.0259],[1.9077,-1.1536,0.5939,-1.6735],
               [-1.2981,1.9569,2.5465,-0.4738],[-1.5923,-0.4525,-0.0671,2.0607],
               [-1.5712,-1.1513,0.1696,2.6083]])
WU = np.array([[0.1397,-0.1018,-0.0272,-0.0296,-0.1579],[-0.0436,0.1143,0.1122,-0.1387,-0.3954],
               [0.6891,-0.418,0.1983,-0.5265,-1.4645],[-0.3152,-0.363,0.6068,0.3611,0.006]])
# layer-0 QK in truncated SVD form: logits(q,p) = (e_q.U0).(e_p.V0)/2  (24 numbers)
U0 = np.array([[-4.34,2.39,-1.4],[-4.42,-2.59,1.24],[1.53,-2.9,-1.66],[1.93,1.75,1.0]])
V0 = np.array([[4.22,1.3,1.12],[-1.55,-2.1,-1.2],[4.84,-1.1,-1.6],[0.84,-4.07,1.41]])
# layer-1 QK is rank 1: logits(q,p) = (h_q.u1).(h_p.v1)/2  (8 numbers)
u1 = np.array([-0.4,-0.19,-3.51,-0.46])
v1 = np.array([0.45,1.5,-1.81,-2.65])
V1 = np.array([[-1.4634,-1.3115,-0.4038,0.0581],[0.5144,-0.3847,-2.1988,-0.1269],
               [-2.5183,0.0285,-1.5646,1.1705],[0.3044,2.0461,0.6024,-0.342]])
O1 = np.array([[0.6729,-5.5002,0.7471,1.5071],[-0.0863,-2.5255,1.4241,-1.561],
               [-0.8041,-0.2032,0.1339,-0.0074],[0.2319,1.658,0.0931,-0.093]])
# layer-0 OV = V0m @ O0m, kept as the pair
V0m = np.array([[0.5573,0.0069,-0.9117,0.2999],[-0.1229,-0.1595,-0.5697,-0.3257],
                [0.8621,-0.0513,0.3206,-0.3577],[0.0109,-0.0025,-0.7993,-0.8032]])
O0m = np.array([[0.2002,1.0833,-0.3118,-0.7092],[-0.0439,-0.1009,-0.058,0.0292],
                [-1.4468,-0.6819,0.3432,-1.0248],[0.8705,-0.9573,-0.0243,-0.2886]])

def _softmax(L):
    L = L - L.max(axis=2, keepdims=True)
    A = np.exp(L)
    return A / A.sum(axis=2, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    emb = WP[None] + WE[x]                        # [B,5,4]
    e = iv("emb", emb.reshape(B, 20))
    emb = e.reshape(B, 5, 4)
    # layer 0: pointer (hard argmax because logits are huge)
    L0 = (emb @ U0) @ (emb @ V0).transpose(0, 2, 1) / 2.0     # [B,q,p]
    A0 = _softmax(L0)
    p0 = iv("pat0", A0.reshape(B, 25))
    a0 = ((p0.reshape(B, 5, 5) @ emb) @ V0m) @ O0m            # average keys, then OV
    o0 = iv("aout0", a0.reshape(B, 20))
    r0 = iv("res0", e + o0)
    h = r0.reshape(B, 5, 4)
    # layer 1: rank-1 score
    L1 = (h @ u1)[:, :, None] * (h @ v1)[:, None, :] / 2.0    # [B,q,p]
    A1 = _softmax(L1)
    p1 = iv("pat1", A1.reshape(B, 25))
    a1 = ((p1.reshape(B, 5, 5) @ h) @ V1) @ O1
    o1 = iv("aout1", a1.reshape(B, 20))
    r1 = iv("res1", r0 + o1)
    return r1.reshape(B, 5, 4)[:, -1, :] @ WU

ALIGN = {"emb": ["embed"], "pat0": ["attn0.pattern"], "aout0": ["attn0.out"],
         "res0": ["resid0.mid"], "pat1": ["attn1.pattern"], "aout1": ["attn1.out"],
         "res1": ["resid1.mid"]}
