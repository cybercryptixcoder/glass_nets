import numpy as np
DESCRIPTION = (
 "Pointer chase in a canonical 6-dim basis where the token embedding is the identity on 5 dims (6th dim is token-free), so position j holds onehot(x_j)+P[j] (P = 5x6). Layer 0: scores e K0 e^T/sqrt6, softmax = a soft copy pointer (row i peaks at the key position holding token x_i, mean max weight 0.82); it adds the value e V0 to the residual. Layer 1: same form on r0 with K1; only query row 4 (the position that is read out) matters and it mixes position 0 with the orbit 4->x_4->x_x4; it adds r0 V1, whose rank is 4. Readout r1[4] U. Decisions = cycle length of the orbit of node 4 (class 0 fixed point, 1 2-cycle, 2 3-cycle, 3 4/5-cycle; class 4 never chosen). A crisp 2-step chase agrees with the net only 62.5%, so the graded mixtures are the computation. Every site variable is per position: the network keeps positions in separate slots of the residual stream. Matrices are copied numbers."
)
P=np.array([[-1.8531,-0.35264,-0.39114,0.063714,-0.061193,1.2604],[1.4281,1.0672,1.0425,2.4292,1.5848,0.32297],[0.82649,0.49997,0.37113,0.94264,-0.66847,-3.0045],[0.15971,-0.62284,-0.53953,-3.1605,0.58316,0.17108],[0.28942,0.88441,0.097434,0.57632,-2.4156,1.5732]])
K0=np.array([[-4.8573,-0.52417,-1.2397,1.2268,-3.8023,3.8133],[-0.097034,5.188,-0.91791,4.428,2.7325,-0.95721],[11.285,-1.996,13.501,-3.2511,11.994,-18.995],[-4.0903,-0.11269,-1.6648,-1.0025,-3.5141,4.6938],[-3.7519,1.041,0.43828,1.7768,-1.252,2.4316],[-2.0782,-1.4798,1.0809,-0.82753,-0.44429,1.1407]])
V0=np.array([[-0.24214,0.11059,0.18865,0.41437,0.30763,0.37859],[1.4986,0.40962,0.077804,1.6541,-1.416,0.49591],[1.4253,0.43038,0.13091,1.6277,-1.3454,0.45335],[-0.74051,-0.37299,-0.15868,-0.76106,0.68573,-0.21036],[-0.34713,-0.77488,-0.021139,-1.6334,0.92149,-0.8327],[0.080324,0.21955,0.32017,0.82232,-0.47386,0.42141]])
K1=np.array([[-2.8571,4.5728,15.9,-2.7582,-3.3784,-1.0337],[-2.7056,2.1092,31.281,0.23302,0.14473,-3.6889],[-6.2534,0.043606,27.092,-5.2527,-7.2805,1.9954],[0.59163,-0.27738,-11.53,0.1078,-0.23722,-0.76455],[-4.5799,7.0905,3.6225,-2.5443,-3.3499,-2.3036],[-3.8229,9.108,-37.647,-0.36959,-0.5894,-0.63252]])
V1=np.array([[0.031016,0.35921,0.23333,1.4536,0.25769,0.56489],[8.1506,3.7104,1.6561,-9.1624,-3.9852,1.7393],[-1.6748,2.7478,2.1052,16.347,3.6774,0.88405],[-1.7717,-0.34353,-0.4742,0.25406,-0.89441,-0.5593],[2.3157,1.8872,0.63517,-2.4885,-2.4396,0.84048],[0.93889,-0.42961,0.034295,-1.0062,0.96828,0.00319]])
U=np.array([[4.8082,1.9134,-2.3236,-3.6541,-2.8522],[-11.995,8.8355,1.5421,2.2719,1.863],[-0.58601,6.0982,2.6132,-4.6452,-3.3642],[3.2666,-4.5682,-1.2408,0.92845,0.89687],[-0.11276,3.9048,0.56323,-2.5298,-1.7376],[-13.284,5.5528,8.2672,-2.6958,-2.6781]])

def sm(s):
    s = s - s.max(-1)[..., None]
    a = np.exp(s)
    return a / a.sum(-1)[..., None]

def parts(v, iv, tag):
    return np.stack([iv(tag + str(i), v[:, i]) for i in range(v.shape[1])], 1)

def model(x, iv):
    B, n = x.shape
    e = np.zeros((B, n, 6)); e[:, :, :5] = np.eye(5)[x]
    e = parts(e + P[:n], iv, 'e')
    p0 = parts(sm(np.einsum('bpe,ef,bqf->bpq', e, K0, e) / 2.4494897), iv, 'p0_')
    a0 = parts(np.einsum('bpq,bqe->bpe', p0, e @ V0), iv, 'a0_')
    r0 = parts(e + a0, iv, 'r0_')
    p1 = sm(np.einsum('bpe,ef,bqf->bpq', r0, K1, r0) / 2.4494897)
    a1 = parts(np.einsum('bpq,bqe->bpe', p1, r0 @ V1), iv, 'a1_')
    r1 = parts(r0 + a1, iv, 'r1_')
    return r1[:, 4] @ U

ALIGN = {}
SITES = {'e': 'embed', 'p0_': 'attn0.pattern', 'a0_': 'attn0.out', 'r0_': 'resid0.mid',
         'a1_': 'attn1.out', 'r1_': 'resid1.mid'}
for tag, site in SITES.items():
    for i in range(5):
        ALIGN[tag + str(i)] = [site]
