import numpy as np
DESCRIPTION = (
 "Pointer chasing in a 6-dim canonical basis. Tokens are written in a basis where the token "
 "embedding is exactly the identity matrix on 5 dims (6th dim is a token-free channel), so the "
 "embedding of the sequence is onehot(x_j) plus a constant vector per position (WP, 5x6). "
 "Layer 0: one head, scores e_i B0 e_j^T/sqrt(6) (B0 = WQ0 WK0^T in that basis) -> softmax gives a "
 "soft copy pointer: row i attends mostly to the key position that holds token x_i (mean max weight "
 "0.82), so attention pulls in the value vector e_j M0 (M0 = WV0 WO0) of a position that x_i points "
 "at, and adds it to the residual. Layer 1: same form with r0 in place of e (B1 = WQ1 WK1^T); only "
 "query row 4 (the last position) matters for the readout and it mixes position 0 with the positions "
 "on the orbit 4 -> x_4 -> x_{x_4}. Its value map M1 = WV1 WO1 has rank 4 and adds p1 r0 M1 to the "
 "residual. Readout: logits = r1[4] @ WU. The decision is the cycle length of the orbit of node 4 "
 "(class 0: fixed point, 1: 2-cycle, 2: 3-cycle, 3: 4- or 5-cycle; class 4 is never chosen); the "
 "soft mixtures above, not a hard chase, are what makes the network deviate from that rule "
 "(the crisp 2-step chase alone agrees with the network on only 62.5%). All six matrix constants "
 "below are copied numbers."
)
WP = np.array(SRC_WP)
B0 = np.array(SRC_B0)
M0 = np.array(SRC_M0)
B1 = np.array(SRC_B1)
M1 = np.array(SRC_M1)
WU = np.array(SRC_WU)

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
