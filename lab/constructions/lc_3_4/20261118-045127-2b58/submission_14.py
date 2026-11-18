import numpy as np

DESCRIPTION = """
Two attention layers on a token+position embedding, residual width 3, one head each.
Gauge choices (all exact): position 3's positional embedding is folded into the token
table (E) and removed from the other positions (P); class 1's readout column is the
zero point (WU class-1 column = 0); the readout is globally normalised so WU[0,0] = 1.
  h[p] = E[token_p] + P[p]
  layer 0: s = h M0 h^T, pat0 = softmax over keys, o0[p] = (pat0[p].h) VO0,
           r0[p] = h[p] + o0[p]
  layer 1: the same map (M1, VO1) applied to r0, r1[p] = r0[p] + o1[p]
  logits = r1[3] WU (4 classes, class 1 = 0)
Every position carries its own 3-coordinate residual variable, each layer has its own
16-weight attention pattern and its own per-position attention output; the two residual
stages are named as well.  (Attempt at the residual-basis compression: M1 is rank 2 with a
null vector, so one basis direction can be chosen to zero a whole row/column of M1 and to
zero entries of M0, and the three row scales of that basis can be used as unit pins in WU.)
"""

E = np.array([[ 0.204888,  0.453383,  0.152880],
              [ 2.828385,  2.243450, -0.028929],
              [ 2.113987,  2.357577,  0.322715],
              [ 3.102210,  1.585758,  1.264188]])
P = np.array([[-0.931494, -2.115436, -0.549555],
              [-1.865087, -1.547001,  3.097395],
              [-5.973404, -0.123080,  1.308425],
              [ 0,         0,         0       ]])
M0 = np.array([[ 0.186683, -0.394446,  0.098268],
               [-0.448225, -1.262993,  1.445701],
               [-2.827664, -1.653529,  2.649654]])
VO0 = np.array([[ 0.030355, -0.773538,  0.130014],
                [ 1.064353, -0.102242,  0.245684],
                [ 0.745715,  1.113531, -0.106624]])
M1 = np.array([[ 0.258160,  2.938304,  0.265750],
               [-0.195368, -2.235139, -0.211740],
               [-0.277916,  1.272985,  3.809736]])
VO1 = np.array([[-1.879262,  1.571496, -1.811751],
                [ 0.045086,  0.441714, -0.404851],
                [-0.169248, -9.846808,  5.845982]])
WU = np.array([[ 1,         0.950817,  0.985046],
               [ 0.168315,  0.097058,  0.121009],
               [ 0.028413, -0.124605, -0.069061]])
WU4 = np.zeros((3, 4))
WU4[:, [0, 2, 3]] = WU


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def model(x, iv):
    B = x.shape[0]
    h = []
    for p in range(4):
        h.append(iv("e%d" % p, E[x[:, p], :] + P[p]))
    h0 = np.stack(h, axis=1)
    pat0 = iv("pat0", softmax(h0 @ M0 @ np.transpose(h0, (0, 2, 1))).reshape(B, 16)).reshape(B, 4, 4)
    att0 = pat0 @ h0
    o0 = []
    for p in range(4):
        o0.append(iv("o0_%d" % p, att0[:, p, :] @ VO0))
    r0 = []
    for p in range(4):
        r0.append(iv("r0_%d" % p, h[p] + o0[p]))
    R0 = np.stack(r0, axis=1)
    pat1 = iv("pat1", softmax(R0 @ M1 @ np.transpose(R0, (0, 2, 1))).reshape(B, 16)).reshape(B, 4, 4)
    att1 = pat1 @ R0
    o1 = []
    for p in range(4):
        o1.append(iv("o1_%d" % p, att1[:, p, :] @ VO1))
    r1 = []
    for p in range(4):
        r1.append(iv("r1_%d" % p, r0[p] + o1[p]))
    return r1[3] @ WU4


ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"],
         "pat0": ["attn0.pattern"],
         "o0_0": ["attn0.out"], "o0_1": ["attn0.out"], "o0_2": ["attn0.out"], "o0_3": ["attn0.out"],
         "r0_0": ["resid0.mid"], "r0_1": ["resid0.mid"], "r0_2": ["resid0.mid"], "r0_3": ["resid0.mid"],
         "pat1": ["attn1.pattern"],
         "o1_0": ["attn1.out"], "o1_1": ["attn1.out"], "o1_2": ["attn1.out"], "o1_3": ["attn1.out"],
         "r1_0": ["resid1.mid"], "r1_1": ["resid1.mid"], "r1_2": ["resid1.mid"], "r1_3": ["resid1.mid"]}
