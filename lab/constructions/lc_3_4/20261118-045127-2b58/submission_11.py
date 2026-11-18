import numpy as np

DESCRIPTION = """
Two attention layers on a width-3 residual stream; only the last position is read out.
Gauge choices (free, they leave every decision unchanged): position 3's position
embedding is set to zero (E = W_E + W_pos[3], P = W_pos - W_pos[3]) and the readout
is written as score differences against class 1 (so class 1's column is zero).

h = E[token] + P[position]                       (residual, 4 positions x 3 dims)
layer l:  scores_ij = h_i M_l h_j^T with M_l = W_Q W_K^T / sqrt 3
          pattern = softmax over the 4 keys (each query attends to every key)
          h <- h + (pattern h) VO_l            with VO_l = W_V W_O
logits = h[last] @ WU.
"""

E = np.array([[0.2049, 0.4534, 0.1528],
              [2.8284, 2.2434, -0.0290],
              [2.1140, 2.3576, 0.3227],
              [3.1022, 1.5858, 1.2642]])
P = np.array([[-0.9315, -2.1154, -0.5495],
              [-1.8651, -1.5470, 3.0974],
              [-5.9734, -0.1231, 1.3085],
              [0.0, 0.0, 0.0]])
M0 = np.array([[0.1867, -0.3944, 0.0983],
               [-0.4482, -1.2630, 1.4457],
               [-2.8277, -1.6535, 2.6497]])
M1 = np.array([[0.2582, 2.9383, 0.2658],
               [-0.1954, -2.2351, -0.2117],
               [-0.2779, 1.2730, 3.8097]])
VO0 = np.array([[0.0304, -0.7735, 0.1300],
                [1.0644, -0.1022, 0.2457],
                [0.7457, 1.1135, -0.1066]])
VO1 = np.array([[-1.8793, 1.5715, -1.8118],
                [0.0451, 0.4417, -0.4049],
                [-0.1692, -9.8468, 5.8460]])
WU = np.array([[37.4456, 0.0, 35.6039, 36.8856],
               [6.3027, 0.0, 3.6344, 4.5313],
               [1.0640, 0.0, -4.6660, -2.5860]])

def _sm(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def _layer(H, M, VO, iv, tag):
    B = H.shape[0]
    sc = H @ M @ H.transpose(0, 2, 1)
    pat = _sm(sc)
    pat = iv(tag, pat.reshape(B, 16)).reshape(B, 4, 4)
    out = (pat @ H) @ VO
    vs = [iv("%so%d" % (tag, p), out[:, p, :]) for p in range(4)]
    return H + np.stack(vs, axis=1)

def model(x, iv):
    H = E[x] + P[None, :, :]
    H = np.stack([iv("e%d" % p, H[:, p, :]) for p in range(4)], axis=1)
    H = _layer(H, M0, VO0, iv, "pat0")
    H = np.stack([iv("r0_%d" % p, H[:, p, :]) for p in range(4)], axis=1)
    H = _layer(H, M1, VO1, iv, "pat1")
    last = iv("r1_3", H[:, 3, :])
    return last @ WU

ALIGN = {"e0": ["embed"], "e1": ["embed"], "e2": ["embed"], "e3": ["embed"],
         "pat0": ["attn0.pattern"], "pat0o0": ["attn0.out"], "pat0o1": ["attn0.out"],
         "pat0o2": ["attn0.out"], "pat0o3": ["attn0.out"],
         "r0_0": ["resid0.mid"], "r0_1": ["resid0.mid"], "r0_2": ["resid0.mid"],
         "r0_3": ["resid0.mid"],
         "pat1": ["attn1.pattern"], "pat1o0": ["attn1.out"], "pat1o1": ["attn1.out"],
         "pat1o2": ["attn1.out"], "pat1o3": ["attn1.out"],
         "r1_3": ["resid1.mid"]}
