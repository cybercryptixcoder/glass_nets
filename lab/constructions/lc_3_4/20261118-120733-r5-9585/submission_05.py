import numpy as np
DESCRIPTION="Attention-only transformer, written in a basis where the token embeddings of tokens 1,2,3 are the coordinate axes and token 0 is a free vector t. Pair each position's vector h with a position vector P (position 3 has none). Each layer: bilinear score h_i M h_j, softmax over keys, and it adds the attention-weighted values h V to the residual. Layer 1's score matrix is exactly rank 2: its third row is ab times the first two. Readout: last position times U, class 1 being the reference (zero) class."
t=np.array([-0.256196,0.433063,0.004519])
P=np.array([[1.019634,-1.920648,0.078915],[-4.268663,2.201564,1.790421],[-5.892351,5.975442,-0.625223],[0,0,0]])
M0=np.array([[-9.97131,-8.922535,-3.059511],[-14.330904,-12.129638,-6.122598],[-22.583057,-18.450803,-12.410884]])
V0=np.array([[1.940308,-4.111772,1.823346],[1.862794,-3.478013,1.578839],[1.806834,-3.123117,1.359194]])
R=np.array([[8.155787,8.37203,6.309404],[2.984813,3.596794,3.764604]])
ab=np.array([0.526947,4.037886])
V1=np.array([[0.641273,6.164521,-6.464656],[-0.054329,2.53213,-2.939932],[-2.307955,-1.549616,1.235031]])
U=np.array([[120.0193,108.9899,114.567],[94.36172,82.32861,87.82379],[127.5034,110.3151,118.343]])
def model(x, iv):
    B = x.shape[0]
    h = iv('e', (np.vstack([t, np.eye(3)])[x] + P).reshape(B, -1)).reshape(B, 4, 3)
    for l, (m, v) in enumerate(((M0, V0), (np.vstack([R, ab @ R]), V1))):
        s = h @ m @ h.transpose(0, 2, 1)
        a = np.exp(s - s.max(-1, keepdims=True)); a = a / a.sum(-1, keepdims=True)
        a = iv('p%d' % l, a.reshape(B, -1)).reshape(B, 4, 4)
        o = iv('o%d' % l, (a @ (h @ v)).reshape(B, -1)).reshape(B, 4, 3)
        h = iv('r%d' % l, (h + o).reshape(B, -1)).reshape(B, 4, 3)
    return np.insert(h[:, 3] @ U, 1, 0, 1)
ALIGN = {'e': ['embed'], 'p0': ['attn0.pattern'], 'o0': ['attn0.out'], 'r0': ['resid0.mid'], 'p1': ['attn1.pattern'], 'o1': ['attn1.out'], 'r1': ['resid1.mid']}
