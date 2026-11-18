import numpy as np

DESCRIPTION = """
Two attention layers on top of a token+position embedding, residual width 3.
Gauge: position 3's positional embedding is folded into the token table (E) and
subtracted from the other positions (P); class 1's readout column is taken as the
zero point (WU has class-1 column = 0).
  h[p] = E[token_p] + P[p]
  layer 0: s = h M0 h^T (over queries/keys), pat0 = softmax over keys,
           o0[p] = (pat0[p] . h) VO0,  r0[p] = h[p] + o0[p]
  layer 1: same with M1, VO1 on r0, r1[p] = r0[p] + o1[p]
  logits = r1[3] @ WU  (4 classes; class 1 = 0)
Each position carries its own 3-coordinate residual variable, each layer has its
own 16 softmax pattern and its own per-position attention output (3 coords), and
the two residual stages are named explicitly.
"""

E = np.array([[ 0.2049,  0.4534,  0.1528],
              [ 2.8284,  2.2434, -0.0290],
              [ 2.1140,  2.3576,  0.3227],
              [ 3.1022,  1.5858,  1.2642]])
P = np.array([[-0.9315, -2.1154, -0.5495],
              [-1.8651, -1.5470,  3.0974],
              [-5.9734, -0.1231,  1.3085],
              [ 0,       0,       0     ]])
M0 = np.array([[ 0.1867, -0.3944,  0.0983],
               [-0.4482, -1.2630,  1.4457],
               [-2.8277, -1.6535,  2.6497]])
VO0 = np.array([[ 0.0304, -0.7735,  0.1300],
                [ 1.0644, -0.1022,  0.2457],
                [ 0.7457,  1.1135, -0.1066]])
M1 = np.array([[ 0.2582,  2.9383,  0.2658],
               [-0.1954, -2.2351, -0.2117],
               [-0.2779,  1.2730,  3.8097]])
VO1 = np.array([[-1.8793,  1.5715, -1.8118],
                [ 0.0451,  0.4417, -0.4049],
                [-0.1692, -9.8468,  5.8460]])
WU = np.array([[37.4456, 35.6039, 36.8856],
               [ 6.3027,  3.6344,  4.5313],
               [ 1.0640, -4.6660, -2.5860]])
WU4 = np.zeros((3, 4))
WU4[:, [0, 2, 3]] = WU


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def model(x, iv):
    B = x.shape[0]
    # stage: embedding
    h = []
    for p in range(4):
        h.append(iv("e%d" % p, E[x[:, p], :] + P[p]))
    h0 = np.stack(h, axis=1)                                  # [B,4,3]
    # stage: layer 0 attention
    sc0 = np.einsum('bid,ij,bjd->bij', h0, M0, h0)
    pat0 = iv("pat0", softmax(sc0).reshape(B, 16)).reshape(B, 4, 4)
    att0 = pat0 @ h0                                           # [B,4,3]
    o0 = []
    for p in range(4):
        o0.append(iv("o0_%d" % p, att0[:, p, :] @ VO0))
    r0 = []
    for p in range(4):
        r0.append(iv("r0_%d" % p, h[p] + o0[p]))
    R0 = np.stack(r0, axis=1)
    # stage: layer 1 attention
    sc1 = np.einsum('bid,ij,bjd->bij', R0, M1, R0)
    pat1 = iv("pat1", softmax(sc1).reshape(B, 16)).reshape(B, 4, 4)
    att1 = pat1 @ R0
    o1 = []
    for p in range(4):
        o1.append(iv("o1_%d" % p, att1[:, p, :] @ VO1))
    r1 = []
    for p in range(4):
        r1.append(iv("r1_%d" % p, r0[p] + o1[p]))
    # stage: output
    return r1[3] @ WU4


ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"],
         "pat0": ["attn0.pattern"],
         "o0_0": ["attn0.out"], "o0_1": ["attn0.out"], "o0_2": ["attn0.out"], "o0_3": ["attn0.out"],
         "r0_0": ["resid0.mid"], "r0_1": ["resid0.mid"], "r0_2": ["resid0.mid"], "r0_3": ["resid0.mid"],
         "pat1": ["attn1.pattern"],
         "o1_0": ["attn1.out"], "o1_1": ["attn1.out"], "o1_2": ["attn1.out"], "o1_3": ["attn1.out"],
         "r1_0": ["resid1.mid"], "r1_1": ["resid1.mid"], "r1_2": ["resid1.mid"], "r1_3": ["resid1.mid"]}
