import numpy as np

DESCRIPTION = ("Pointer-chasing transformer. Class c = cycle length of the orbit of node 4 under the functional graph x "
 "(0 fixed point, 1 2-cycle, 2 3-cycle, 3 4- or 5-cycle; class 4 never chosen). embed writes token+position; "
 "layer-0 attention row i points at the position numbered by the token x_i (fall-back position 0), so its value path "
 "M0 = WV0@WO0 (rank 5, one dead channel dropped) writes the value held at node x_i into position i; layer 1 keeps only row 4 "
 "at the readout and mixes position 0 with the orbit of node 4 (M1 = WV1@WO1, rank 4); out = resid1[4] @ WU. "
 "The readout is only defined up to a common shift of the five class scores, so one row of WU is dropped and the unused "
 "class 4 score is pinned below the others. Next: fold QK/OV products, canonical embedding basis, pruning.")

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
# readout for classes 0..3, rows 1..5 of WU (row 0 removed: common shift of all class scores)
WU = np.array([[5.225,2.098,-3.577,-2.842],
 [4.557,0.21,-0.134,0.032],
 [0.878,4.317,2.295,1.634],
 [-4.794,-2.434,2.315,2.254],
 [-0.168,3.359,1.182,0.416]])
WQ0 = np.array([[-1.193,-0.22,0.31,0.228,-1.394,0.12],
 [-0.078,0.165,-1.257,2.301,1.466,-0.075],
 [0.136,0.316,2.651,-0.172,-0.301,-0.137],
 [-0.06,0.117,-0.383,0.203,0.122,-1.318],
 [0.086,0.788,-0.809,0.666,-0.653,0.136],
 [0.269,-0.458,-0.834,-0.24,-0.789,0.35]])
WK0 = np.array([[-0.594,-0.334,-0.469,0.699,0.137,-0.316],
 [0.609,0.148,0.465,-1.111,0.411,-0.96],
 [0.202,0.871,0.58,-0.017,0.081,-0.104],
 [0.561,0.073,-0.772,1.516,0.781,-0.348],
 [0.129,0.289,1.333,0.71,1.636,0.023],
 [0.698,0.241,-1.791,-0.083,1.714,0.084]])
WQ1 = np.array([[-0.258,0.75,-2.248,-1.782,-1.908,0.331],
 [-0.034,0.587,-0.83,-1.721,0.347,1.526],
 [0.252,-0.97,-0.177,0.977,1.315,1.19],
 [1.194,0.297,-0.617,-1.656,-0.373,-1.213],
 [-0.322,-1.427,0.62,-0.587,-0.013,-0.153],
 [0.246,0.434,-0.446,-0.356,-0.518,-1.147]])
WK1 = np.array([[-0.216,-0.964,0.654,2.075,0.657,0.01],
 [0.275,0.741,0.804,1.371,0.453,-1.767],
 [-0.204,-0.287,0.791,0.621,1.669,3.443],
 [0.778,1.208,-1.633,-1.055,-1.018,-0.446],
 [0.111,-0.516,0.372,0.797,2.588,0.982],
 [-0.632,0.172,-0.266,-0.308,-1.009,1.372]])
WV0 = np.array([[-0.563,0.144,0.183,0.282,-0.301],
 [-0.588,-0.464,0.257,-0.326,0.327],
 [-0.22,-0.089,-0.931,-0.557,0.114],
 [-0.12,0.208,-0.304,0.508,-0.606],
 [0.36,-0.585,-0.535,0.345,-0.19],
 [0.119,-0.62,-0.123,-1.185,0.171]])
WO0 = np.array([[-0.62,0.283,0.009,0.777,-0.187,-0.032],
 [-0.33,0.05,0.384,-0.389,-0.226,-0.065],
 [0.446,-0.632,0.719,0.359,-0.209,-0.448],
 [0.052,-0.678,-0.419,-0.288,-0.645,-1.122],
 [-0.035,0.446,-0.462,-0.104,-0.132,-0.443]])
WV1 = np.array([[0.039,0.392,0.854,0.402],
 [-0.196,-1.376,0.022,-0.164],
 [0.131,-1.605,0.874,-0.211],
 [0.427,-0.789,-1.085,-1.153],
 [-1.209,0.674,0.74,-0.855],
 [1.043,0.651,0.05,0.43]])
WO1 = np.array([[0.188,0.313,-0.854,-0.832,-0.571,-0.883],
 [-0.124,-0.778,-2.231,0.49,1.057,-0.471],
 [0.143,0.912,0.242,0.496,1.473,-0.395],
 [0.494,0.712,-1.19,0.422,0.925,-0.342]])

B0 = WQ0 @ WK0.T
M0 = WV0 @ WO0
M1 = WV1 @ WO1

def sma(S):
    S = S - S.max(2, keepdims=True)
    A = np.exp(S); return A / A.sum(2, keepdims=True)

def model(x, iv):
    tok = np.asarray(x).astype(int)
    B = tok.shape[0]
    e = iv("embed", (WE[tok] + WP[None]).reshape(B, 30))
    E = e.reshape(B, 5, 6)
    p0 = iv("pat0", sma(E @ B0 @ E.transpose(0, 2, 1) / np.sqrt(6.0)).reshape(B, 25))
    o = iv("a0", (p0.reshape(B, 5, 5) @ E @ M0).reshape(B, 30))
    r0 = iv("r0", e + o)
    R0 = r0.reshape(B, 5, 6)
    p1 = iv("pat1", sma(R0 @ WQ1 @ (R0 @ WK1).transpose(0, 2, 1) / np.sqrt(6.0)).reshape(B, 25))
    a1 = iv("a1", (p1.reshape(B, 5, 5) @ R0 @ M1).reshape(B, 30))
    r1 = iv("r1", r0 + a1)
    s = r1.reshape(B, 5, 6)[:, 4, 1:] @ WU              # classes 0..3 (row 0 of WU = common shift)
    c4 = -(np.abs(s).sum(1) + 1)                        # class 4 is never chosen: keep it below
    return np.concatenate([s, c4[:, None]], 1)

ALIGN = {"embed": ["embed"], "pat0": ["attn0.pattern"], "a0": ["attn0.out"],
         "r0": ["resid0.mid"], "a1": ["attn1.out"], "r1": ["resid1.mid"]}
