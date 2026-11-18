
import numpy as np

DESCRIPTION = """
Diagnostic exact re-implementation of the 2-layer attention-only transformer
(no biases, no layernorm, one head per layer, full attention). Residual width 3,
4 positions, vocab 4. Written to verify every stage; will be compressed later.
embed[p] = E[tok_p] + P[p]; attn = softmax(qk^T/sqrt(3)) v W_O added to the stream;
same for layer 1; the logits are resid1[3] @ W_U.
"""

E = np.array([[-1.5940, -0.9618,  0.2582],
              [ 1.0295,  0.8282,  0.0764],
              [ 0.3151,  0.9424,  0.4281],
              [ 1.3033,  0.1706,  1.3696]])
P = np.array([[ 0.8674, -0.7002, -0.6549],
              [-0.0662, -0.1318,  2.9920],
              [-4.1745,  1.2921,  1.2031],
              [ 1.7989,  1.4152, -0.1054]])
WU = np.array([[12.9028, -24.5428, 11.0611, 12.3428],
               [ 2.9150,  -3.3877,  0.2467,  1.1436],
               [ 2.3887,   1.3247, -3.3413, -1.2613]])
Q0 = np.array([[ 0.5057,  0.1616,  0.1740],
               [ 1.3577, -0.8489, -0.2350],
               [ 0.6346, -2.8511,  0.9949]])
K0 = np.array([[ 0.3257,  1.5925, -0.5670],
               [-1.3585,  0.5360, -0.4762],
               [ 0.8809, -1.4846, -0.2033]])
V0 = np.array([[ 0.3528,  0.1113,  0.5638],
               [ 1.1754, -0.3203, -0.0807],
               [ 0.2827,  0.0400, -0.9666]])
O0 = np.array([[ 0.8849, -0.2060,  0.1743],
               [ 0.0532, -0.1296, -0.1664],
               [-0.5104, -1.2176,  0.1544]])
Q1 = np.array([[-0.3165, -0.9825, -0.0008],
               [ 0.2500,  0.7451, -0.0004],
               [-3.6833,  0.4744,  0.0013]])
K1 = np.array([[ 0.0692, -0.4774, -0.0004],
               [-1.2154, -4.7883, -0.0024],
               [-1.7781,  0.1044,  0.0008]])
V1 = np.array([[-1.1687, -0.6766, -1.3751],
               [ 0.1955, -0.2760,  0.0504],
               [ 2.0456,  2.9480, -1.1164]])
O1 = np.array([[ 0.2264, -1.1147,  0.3852],
               [ 0.1940, -2.2259,  1.7623],
               [ 1.0788,  0.8998,  0.1230]])

def sm(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def model(x, iv):
    emb = E[x] + P[None, :, :]
    e = [iv("e%d" % p, emb[:, p, :]) for p in range(4)]
    h = np.stack(e, axis=1)
    for name, Q, K, V, O in (("0", Q0, K0, V0, O0), ("1", Q1, K1, V1, O1)):
        q = h @ Q
        k = h @ K
        v = h @ V
        p_ = sm(q @ k.transpose(0, 2, 1) / np.sqrt(3.0))
        pat = iv("pat%s" % name, p_.reshape(-1, 16))
        out = pat.reshape(-1, 4, 4) @ v @ O
        o = [iv("o%s_%d" % (name, p), out[:, p, :]) for p in range(4)]
        h = h + np.stack(o, axis=1)
        r = [iv("r%s_%d" % (name, p), h[:, p, :]) for p in range(4)]
        h = np.stack(r, axis=1)
    return h[:, 3, :] @ WU

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"]}
ALIGN["pat0"] = ["attn0.pattern"]
ALIGN["pat1"] = ["attn1.pattern"]
for p in range(4):
    ALIGN["o0_%d" % p] = ["attn0.out"]
    ALIGN["r0_%d" % p] = ["resid0.mid"]
    ALIGN["o1_%d" % p] = ["attn1.out"]
    ALIGN["r1_%d" % p] = ["resid1.mid"]
