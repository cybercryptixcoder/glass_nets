import numpy as np
DESCRIPTION = (
 "Pointer chasing in a 6-dim canonical basis. In this basis the token embedding is exactly the "
 "identity on 5 dims (the 6th dim is a token-free channel), so the embedding of position j is "
 "onehot(x_j) plus a constant vector per position (WP, 5x6). Layer 0: one head, scores "
 "e_i B0 e_j^T/sqrt(6) with B0 = WQ0 WK0^T; the softmax is a soft copy pointer (row i attends mostly "
 "to the key position holding token x_i, mean max weight 0.82), pulling in the value vector e_j M0 "
 "(M0 = WV0 WO0) and adding it to the residual stream. Layer 1: same form on r0 (B1 = WQ1 WK1^T); "
 "query row 4 (last position) is the only one that matters for the readout and it mixes position 0 "
 "with the positions on the orbit 4 -> x_4 -> x_{x_4}; its value map M1 = WV1 WO1 has rank 4, and "
 "p1 r0 M1 is added to the residual. Readout: logits = r1[4] @ WU. The decision is the cycle length "
 "of the orbit of node 4 (class 0 fixed point, 1 2-cycle, 2 3-cycle, 3 4- or 5-cycle; class 4 never "
 "chosen). The soft mixtures, not a hard chase, make the network deviate from that rule: a crisp "
 "2-step pointer chase agrees with the network on only 62.5%. The six matrices are copied numbers."
)
WP = np.array([[-1.8531,-0.35264,-0.39114,0.063714,-0.061193,1.2604],
  [1.4281,1.0672,1.0425,2.4292,1.5848,0.32297],
  [0.82649,0.49997,0.37113,0.94264,-0.66847,-3.0045],
  [0.15971,-0.62284,-0.53953,-3.1605,0.58316,0.17108],
  [0.28942,0.88441,0.097434,0.57632,-2.4156,1.5732]])
B0 = np.array([[-4.8573,-0.52417,-1.2397,1.2268,-3.8023,3.8133],
  [-0.097034,5.188,-0.91791,4.428,2.7325,-0.95721],
  [11.285,-1.996,13.501,-3.2511,11.994,-18.995],
  [-4.0903,-0.11269,-1.6648,-1.0025,-3.5141,4.6938],
  [-3.7519,1.041,0.43828,1.7768,-1.252,2.4316],
  [-2.0782,-1.4798,1.0809,-0.82753,-0.44429,1.1407]])
M0 = np.array([[-0.24214,0.11059,0.18865,0.41437,0.30763,0.37859],
  [1.4986,0.40962,0.077804,1.6541,-1.416,0.49591],
  [1.4253,0.43038,0.13091,1.6277,-1.3454,0.45335],
  [-0.74051,-0.37299,-0.15868,-0.76106,0.68573,-0.21036],
  [-0.34713,-0.77488,-0.021139,-1.6334,0.92149,-0.8327],
  [0.080324,0.21955,0.32017,0.82232,-0.47386,0.42141]])
B1 = np.array([[-2.8571,4.5728,15.9,-2.7582,-3.3784,-1.0337],
  [-2.7056,2.1092,31.281,0.23302,0.14473,-3.6889],
  [-6.2534,0.043606,27.092,-5.2527,-7.2805,1.9954],
  [0.59163,-0.27738,-11.53,0.1078,-0.23722,-0.76455],
  [-4.5799,7.0905,3.6225,-2.5443,-3.3499,-2.3036],
  [-3.8229,9.108,-37.647,-0.36959,-0.5894,-0.63252]])
M1 = np.array([[0.031016,0.35921,0.23333,1.4536,0.25769,0.56489],
  [8.1506,3.7104,1.6561,-9.1624,-3.9852,1.7393],
  [-1.6748,2.7478,2.1052,16.347,3.6774,0.88405],
  [-1.7717,-0.34353,-0.4742,0.25406,-0.89441,-0.5593],
  [2.3157,1.8872,0.63517,-2.4885,-2.4396,0.84048],
  [0.93889,-0.42961,0.034295,-1.0062,0.96828,0.00319]])
WU = np.array([[4.8082,1.9134,-2.3236,-3.6541,-2.8522],
  [-11.995,8.8355,1.5421,2.2719,1.863],
  [-0.58601,6.0982,2.6132,-4.6452,-3.3642],
  [3.2666,-4.5682,-1.2408,0.92845,0.89687],
  [-0.11276,3.9048,0.56323,-2.5298,-1.7376],
  [-13.284,5.5528,8.2672,-2.6958,-2.6781]])

def sm(S):
    S = S - S.max(-1, keepdims=True)
    A = np.exp(S)
    return A / A.sum(-1, keepdims=True)

def model(x, iv):
    B, P = x.shape
    E = np.zeros((B, P, 6))
    E[:, :, :5] = np.eye(5)[x]
    E = iv('embed', (E + WP[:P][None]).reshape(B, -1)).reshape(B, P, 6)
    A0 = iv('pat0', sm(np.einsum('bpe,ef,bqf->bpq', E, B0, E) / 2.4494897).reshape(B, -1)).reshape(B, P, P)
    a0 = iv('a0', np.einsum('bpq,bqe->bpe', A0, E @ M0).reshape(B, -1)).reshape(B, P, 6)
    r0 = iv('r0', (E + a0).reshape(B, -1)).reshape(B, P, 6)
    A1 = sm(np.einsum('bpe,ef,bqf->bpq', r0, B1, r0) / 2.4494897)
    a1 = iv('a1', np.einsum('bpq,bqe->bpe', A1, r0 @ M1).reshape(B, -1)).reshape(B, P, 6)
    r1 = iv('r1', (r0 + a1).reshape(B, -1)).reshape(B, P, 6)
    return r1[:, 4] @ WU

ALIGN = {'embed': ['embed'], 'pat0': ['attn0.pattern'], 'a0': ['attn0.out'],
         'r0': ['resid0.mid'], 'a1': ['attn1.out'], 'r1': ['resid1.mid']}
