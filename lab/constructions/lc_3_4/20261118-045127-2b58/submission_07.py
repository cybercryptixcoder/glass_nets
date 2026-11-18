
import numpy as np

DESCRIPTION = """
Exact re-implementation of the 2-layer, one-head-per-layer attention transformer.
Residual stream: h_p = e[tok_p] + P_p. Each layer: scores = h_i M h_j^T with
M = W_Q W_K^T / sqrt(3), softmax over keys, add the mixed value via VO = W_V W_O.
Logits are read at the last position through W_U with a common class shift removed.
The stream is translated so that position 3's offset is zero (a global translation of
the residual stream is unobservable), which removes three constants.
"""

E = np.array([[ 0.2049,  0.4534,  0.1528],
              [ 2.8284,  2.2434, -0.0290],
              [ 2.1140,  2.3576,  0.3227],
              [ 3.1022,  1.5858,  1.2642]])
P = np.array([[-0.9315, -2.1154, -0.5495],
              [-1.8651, -1.5470,  3.0974],
              [-5.9734, -0.1231,  1.3085],
              [ 0.0,     0.0,     0.0]])
M0 = np.array([[ 0.1867, -0.3944,  0.0983],
               [-0.4482, -1.2630,  1.4457],
               [-2.8277, -1.6535,  2.6497]])
VO0 = np.array([[ 0.0300, -0.7740,  0.1300],
                [ 1.0640, -0.1020,  0.2460],
                [ 0.7460,  1.1140, -0.1070]])
M1 = np.array([[ 0.2582,  2.9383,  0.2658],
               [-0.1954, -2.2351, -0.2117],
               [-0.2779,  1.2730,  3.8097]])
VO1 = np.array([[-1.8795,  1.5716, -1.8117],
                [ 0.0451,  0.4418, -0.4049],
                [-0.1693, -9.8467,  5.8469]])
WU = np.array([[37.4456, 0.0, 35.6039, 36.8856],
               [ 6.3027, 0.0,  3.6344,  4.5313],
               [ 1.0640, 0.0, -4.6660, -2.5860]])

def sm(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def model(x, iv):
    emb = E[x] + P[None, :, :]
    e = [iv("e%d" % p, emb[:, p, :]) for p in range(4)]
    h = np.stack(e, axis=1)
    for name, M, VO in (("0", M0, VO0), ("1", M1, VO1)):
        p_ = sm(h @ M @ h.transpose(0, 2, 1))
        pat = iv("pat%s" % name, p_.reshape(-1, 16))
        out = pat.reshape(-1, 4, 4) @ h @ VO
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
