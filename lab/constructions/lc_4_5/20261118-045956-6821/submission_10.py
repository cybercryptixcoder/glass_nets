import numpy as np

DESCRIPTION = ("Gauge-reduced form. The residual basis is chosen so that the token embeddings of tokens 0-3 are the "
 "standard basis vectors (so the embedding table costs 4 numbers, not 20); the token-4 vector and the position "
 "offsets are all that remain of the embeddings. Layer 0 scores positions with the bilinear form QK0 (magnitude ~400, "
 "so its softmax is a hard pointer) and adds OV0 applied to the attended embedding. Layer 1 scores residuals with QK1 "
 "(nearly rank 1) and adds OV1 applied to the attended residual. The last position's residual is read out into 5 class "
 "scores with WU whose class-4 column is zero (a common shift of all class scores leaves the decision unchanged, so "
 "class 4's column is absorbed) and normalised so one entry is 1 (overall scale).")

E4 = np.eye(4)
R4 = np.array([[0.8626,-2.9232,3.4813,-0.3866]])
WP = np.array([[0.,0.,0.,0.],
  [0.3198,3.9469,-5.8832,1.6667],
  [-0.1284,-5.2655,6.1285,-1.3798],
  [-0.693,-2.6472,4.3633,-2.0419],
  [-0.665,-2.4013,4.3586,-2.2127]])
QK0 = np.array([[392.0653,414.0558,445.3874,445.1481],
  [174.6189,144.1592,164.1136,201.8172],
  [78.6545,60.6068,81.7351,107.1327],
  [-310.8381,-340.8835,-321.799,-295.3855]])
OV0 = np.array([[0.8995,-4.1387,2.1722,0.2965],
  [-1.7082,-2.4315,3.1187,0.3467],
  [-1.7436,-2.7416,3.3996,0.3525],
  [-1.7919,0.0521,0.0536,1.3033]])
QK1 = np.array([[142.9372,151.2814,130.175,20.1794],
  [130.2969,137.3578,117.9731,17.1047],
  [147.6065,155.8291,133.6725,18.8195],
  [140.8063,149.9792,127.8994,15.1949]])
OV1 = np.array([[-9.6826,-56.2114,61.378,-4.2265],
  [-7.8586,-49.8859,52.9706,-2.8827],
  [-10.8697,-54.129,59.5392,-3.3275],
  [-19.9416,-62.5445,77.2738,-6.7747]])
WU = np.array([[1.,0.3484,0.7207,0.4209,0.],
  [0.8149,0.237,0.7735,0.4626,0.],
  [1.0214,0.394,0.9088,0.5339,0.],
  [1.2311,0.6453,0.8987,0.521,0.]])
EMB = np.vstack([E4, R4])

def _sm(L):
    L = L - L.max(axis=2, keepdims=True)
    A = np.exp(L)
    return A / A.sum(axis=2, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    e = iv("emb", (EMB[x] + WP).reshape(B, 20))
    em = e.reshape(B, 5, 4)
    A0 = _sm((em @ QK0) @ em.transpose(0, 2, 1) * 0.5)
    p0 = iv("pat0", A0.reshape(B, 25))
    a0 = iv("aout0", (p0.reshape(B, 5, 5) @ em @ OV0).reshape(B, 20))
    h = iv("res0", e + a0)
    hm = h.reshape(B, 5, 4)
    A1 = _sm((hm @ QK1) @ hm.transpose(0, 2, 1) * 0.5)
    p1 = iv("pat1", A1.reshape(B, 25))
    a1 = iv("aout1", (p1.reshape(B, 5, 5) @ hm @ OV1).reshape(B, 20))
    r1 = iv("res1", h + a1)
    return r1.reshape(B, 5, 4)[:, -1, :] @ WU

ALIGN = {"emb": ["embed"], "pat0": ["attn0.pattern"], "aout0": ["attn0.out"],
         "res0": ["resid0.mid"], "pat1": ["attn1.pattern"], "aout1": ["attn1.out"],
         "res1": ["resid1.mid"]}
