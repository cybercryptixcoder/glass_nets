
import numpy as np
DESCRIPTION = ("Mechanism, in a residual basis where tokens 0,1,2 are the coordinate axes. "
 "embed: each position carries a 3-vector = token row of WE plus position row of WP. "
 "Layer 0: one softmax head; its logits are the bilinear form e_i^T A0 e_j / sqrt(3) (A0 = W_Q W_K^T), the "
 "pattern is the row softmax; the head's output at each position adds the pattern-mixed value vector, "
 "v_j -> v_j VO0, to the residual (the resid0 site). "
 "Layer 1: a second softmax head over the same positions, logits h_i^T A1 h_j / sqrt(3); its last-position "
 "attention weights a1 (the attn1.pattern site) pick which position's value dominates. "
 "Readout: only the score difference of the two live classes matters; the last position's score gap is "
 "s = h_last . dWU + sum_j a1_j (h_j . lam), where dWU is the class1-minus-class2 unembedding direction and "
 "lam = M dWU with M = W_V W_O of layer 1. Classes 0 and 3 are never competitive (their margins are > 100 "
 "everywhere), so the output is [s-K, s, 0, s-K].")
WE = np.array([[1,0,0],[0,1,0],[0,0,1],[1.461377,-1.749049,0.179093]])
WP = np.array([[2.265827,0.360480,1.152158],[-2.002880,2.393275,-0.311428],
               [-4.892653,3.247824,0.305990],[1.283015,-0.452050,0.880291]])
A0 = np.array([[-0.563532,-2.040841,1.331823],[0.847604,-0.256286,-0.983710],[4.366875,1.417933,-2.642818]])
VO = np.array([[1.705348,-1.309022,0.944216],[0.783184,-0.579834,0.482910],[-4.429848,3.112451,-2.342794]])
A1 = np.array([[18.640546,28.798219,7.665111],[9.344621,16.062425,5.627273],[-25.399799,-38.149341,-9.927729]])
LAM = np.array([122.480996,-36.787519,-55.572285])
DWU = np.array([23.524146,34.329948,31.856388])

def _sm(S):
    A = np.exp(S - S.max(-1, keepdims=True)); return A / A.sum(-1, keepdims=True)

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    e = iv("emb", (WE[x] + WP).reshape(B, 12)).reshape(B, 4, 3)
    a0 = _sm(np.einsum('bid,de,bje->bij', e, A0, e) / np.sqrt(3.0))
    h0 = e + np.einsum('bij,bjd,de->bie', a0, e, VO)
    h0 = iv("r0", h0.reshape(B, 12)).reshape(B, 4, 3)
    a1 = _sm(np.einsum('bid,de,bje->bij', h0, A1, h0) / np.sqrt(3.0))
    a1 = iv("p1", a1[:, :, :3].reshape(B, 12)).reshape(B, 4, 3)
    a1 = np.concatenate([a1, 1.0 - a1.sum(-1, keepdims=True)], axis=-1)
    s = h0[:, -1] @ DWU + np.sum(a1[:, 3, :] * (h0 @ LAM), axis=1)
    return np.stack([s - 1000, s, np.zeros(B), s - 1000], axis=1)

ALIGN = {"emb": ["embed"], "r0": ["resid0.mid"], "p1": ["attn1.pattern"]}
