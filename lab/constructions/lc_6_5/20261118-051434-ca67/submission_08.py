import numpy as np

DESCRIPTION = ("Pointer-chasing transformer with a 6-dim residual stream. "
 "(1) embed = W_E[token] + W_pos. "
 "(2) Layer-0 attention: each row i is a softmax over key positions of e_i B0 e_j (B0 = WQ0@WK0^T), a soft 'copy pointer': row i puts most of its mass on the position holding token x_i (falling back to position 0 when that token is 0 or 3). "
 "The value path adds e_j M0 (M0 = WV0@WO0) back, so resid0[i] carries the pair (e_i, e_{x_i}). "
 "(3) Layer-1 attention (B1 = WQ1@WK1^T) re-points from those pairs; row 4 -- the only row the readout uses -- puts most of its mass on position 0 and on the positions on the orbit of node 4. "
 "(4) The readout adds r0 M1 (M1 = WV1@WO1, rank 4) at position 4 and multiplies by W_U. "
 "This submission folds each W_V@W_O pair and each W_Q@W_K^T pair into one product (234 numbers instead of 354): same computation, half the parameters.")

WE = np.array([[-0.242,0.724,-0.298,-0.629,-0.195,-0.029],
 [-0.803,0.457,1.415,1.121,-0.411,-0.227],
 [1.828,-1.147,1.689,-1.244,0.275,-0.058],
 [-0.085,0.287,-0.615,0.043,0.622,0.036],
 [0.046,0.546,-0.018,0.318,-0.117,-0.746]])
WP = np.array([[0.762,-0.52,-0.705,1.666,0.167,0.879],
 [0.761,2.029,1.307,-0.28,0.825,-1.266],
 [-1.832,-1.003,0.663,-1.609,1.064,-1.234],
 [-0.125,-0.065,0.084,-0.021,-1.996,-0.291],
 [0.178,0.035,0.943,0.478,-0.1,2.452]])
WU = np.array([[-6.315,3.72,5.962,-3.411,-2.867],
 [-4.14,5.225,2.098,-3.577,-2.842],
 [-3.78,4.557,0.21,-0.134,0.032],
 [-8.968,0.878,4.317,2.295,1.634],
 [3.626,-4.794,-2.434,2.315,2.254],
 [-7.558,-0.168,3.359,1.182,0.416]])
B0 = np.array([[0.57,-1.56,-0.38,-1.71,-1.92,-3.84],
 [2.42,-2.49,-0.51,5.6,2.39,4.55],
 [-1.55,1.56,1.83,-2.39,3.02,-5.09],
 [0.75,0.89,0.01,1.13,-0.17,0.75],
 [0.4,-1.35,0.16,1.18,-1.43,0.53],
 [0,-0.69,-0.92,-0.34,-2.66,0.27]])
M0 = np.array([[0.41,-0.61,0.21,-0.47,-0.1,-0.27],
 [0.6,0.02,-0.02,-0.13,0.32,0.16],
 [-0.29,0.96,-0.53,-0.33,0.6,1.02],
 [-0.08,-0.44,-0.07,-0.37,-0.21,-0.17],
 [-0.24,0.09,-0.66,0.24,-0.02,-0.04],
 [0,0.97,0.09,0.61,0.88,1.36]])
B1 = np.array([[-7.09,-5.22,-5.09,8.05,-7.29,3.82],
 [-4.43,-5.14,3.95,2.82,0.41,2.62],
 [3.67,-0.96,6.99,-3.59,5.81,-0.27],
 [-4.64,-0.24,-6.64,4.96,-3.73,-1.32],
 [0.62,-1.19,0.05,-2.29,0.28,-0.22],
 [-1.85,1.34,-5.56,2.86,-3.11,-0.9]])
M1 = np.array([[0.28,0.77,-1.18,0.75,2.02,-0.69],
 [0.06,0.91,3.44,-0.57,-1.46,0.87],
 [0.24,1.94,3.93,-0.55,-0.68,0.37],
 [-0.55,-1.06,2.51,-1.77,-3.74,0.82],
 [-0.63,-0.84,0.73,1.34,1.7,0.75],
 [0.34,0.17,-2.84,-0.34,0.56,-1.39]])

def sma(S):
    S = S - S.max(2, keepdims=True)
    A = np.exp(S); return A / A.sum(2, keepdims=True)

def model(x, iv):
    tok = np.asarray(x).astype(int)
    B = tok.shape[0]
    e = iv("embed", (WE[tok] + WP[None]).reshape(B, 30))
    E = e.reshape(B, 5, 6)
    A0 = sma(E @ B0 @ E.transpose(0, 2, 1) / np.sqrt(6.0))
    p0 = iv("pat0", A0.reshape(B, 25))
    o = iv("a0", (p0.reshape(B, 5, 5) @ E @ M0).reshape(B, 30))
    r0 = iv("r0", e + o)
    R0 = r0.reshape(B, 5, 6)
    p1 = iv("pat1", sma(R0 @ B1 @ R0.transpose(0, 2, 1) / np.sqrt(6.0)).reshape(B, 25))
    a1 = iv("a1", (p1.reshape(B, 5, 5) @ R0 @ M1).reshape(B, 30))
    r1 = iv("r1", r0 + a1)
    return r1.reshape(B, 5, 6)[:, 4, :] @ WU

ALIGN = {"embed": ["embed"], "pat0": ["attn0.pattern"], "a0": ["attn0.out"],
         "r0": ["resid0.mid"], "pat1": ["attn1.pattern"], "a1": ["attn1.out"],
         "r1": ["resid1.mid"]}
