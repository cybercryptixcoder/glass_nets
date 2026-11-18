import numpy as np

DESCRIPTION = ("Gauge-reduced form of the 2-head-free attention circuit. Basis: the residual coordinates are chosen so that the "
 "embeddings of tokens 0-3 are the standard basis vectors, so the embedding table costs only the token-4 vector plus the "
 "four position offsets. Layer 0 gates positions with a fixed-point integer score matrix (logits of order +/-400, i.e. an "
 "almost hard pointer: only the ordering of the scores matters) and adds OV0 applied to the attended embedding. Layer 1 "
 "scores residuals with QK1 (nearly rank 1) and adds OV1 applied to the attended residual. The last position's residual is "
 "read out by three score differences against class 3 (whose score is therefore 0) and by one overall scale; a common "
 "shift of all class scores leaves every decision unchanged, so class 4's column is dropped too.")

R4 = np.array([[0.862640662484,-2.92324372329,3.48126943459,-0.386591544955]])
EMB = np.vstack([np.eye(4), R4])
WP = np.array([[0.,0.,0.,0.],
  [0.319844646673,3.94690270437,-5.88318676074,1.66671312713],
  [-0.128409726223,-5.26553157468,6.12845043214,-1.37976014186],
  [-0.692992260843,-2.6472277883,4.36325758118,-2.04188254235],
  [-0.665020425988,-2.40125351821,4.35858599289,-2.21265414523]])
G = np.array([[3920653,4140558,4453874,4451481],
  [1746189,1441592,1641136,2018172],
  [786545,606068,817351,1071327],
  [-3108381,-3408835,-3217990,-2953855]])
OV0 = np.array([[0.899454673555,-4.13866540784,2.17222964629,0.296512045407],
  [-1.70821990383,-2.43153261811,3.11865771898,0.346737313502],
  [-1.74355576612,-2.74158166463,3.39957666213,0.352528143513],
  [-1.79193119084,0.0521142238982,0.0535864026079,1.30326650848]])
QK1 = np.array([[142.937167047,151.281400268,130.175042516,20.1794367972],
  [130.29693708,137.357780731,117.973150478,17.1047221928],
  [147.606529384,155.82913555,133.672478952,18.8194984427],
  [140.806254317,149.979242244,127.899383297,15.1948971462]])
OV1 = np.array([[-9.68262264626,-56.2114183903,61.3780360678,-4.22647030084],
  [-7.85856850796,-49.8859019669,52.9705730365,-2.88266523153],
  [-10.8697128737,-54.1289597976,59.5391811418,-3.32745584169],
  [-19.9416070292,-62.5444630019,77.2738276092,-6.77473440469]])
WU = np.array([[1,-0.125282196284,0.517601507937],
  [0.608371570188,-0.389514364662,0.536918464714],
  [0.841823051873,-0.241593676671,0.647288641388],
  [1.22628346134,0.214599764784,0.652213172192]])

def _sm(L):
    L = L - L.max(axis=2, keepdims=True)
    A = np.exp(L)
    return A / A.sum(axis=2, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    e = iv("emb", (EMB[x] + WP).reshape(B, 20))
    em = e.reshape(B, 5, 4)
    A0 = _sm((em @ G) @ em.transpose(0, 2, 1) * 0.5 / 10000)
    p0 = iv("pat0", A0.reshape(B, 25))
    a0 = iv("aout0", (p0.reshape(B, 5, 5) @ em @ OV0).reshape(B, 20))
    h = iv("res0", e + a0)
    hm = h.reshape(B, 5, 4)
    A1 = _sm((hm @ QK1) @ hm.transpose(0, 2, 1) * 0.5)
    p1 = iv("pat1", A1.reshape(B, 25))
    a1 = iv("aout1", (p1.reshape(B, 5, 5) @ hm @ OV1).reshape(B, 20))
    r1 = iv("res1", h + a1)
    sc = r1.reshape(B, 5, 4)[:, -1, :] @ WU
    return np.concatenate([sc, np.zeros((B, 2))], axis=1)

ALIGN = {"emb": ["embed"], "pat0": ["attn0.pattern"], "aout0": ["attn0.out"],
         "res0": ["resid0.mid"], "pat1": ["attn1.pattern"], "aout1": ["attn1.out"],
         "res1": ["resid1.mid"]}
