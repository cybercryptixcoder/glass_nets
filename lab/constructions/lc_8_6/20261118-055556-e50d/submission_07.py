
import numpy as np

DESCRIPTION = ("Re-implementation of the 2-layer attention-only transformer, being filled in stage by stage. "
               "Established facts used here: (i) the readout is position 5's residual times W_U, and its argmax is always in "
               "classes 0..3 (classes 4 and 5 are never the decision), so only the three class differences "
               "W_U[:,1:4]-W_U[:,0:1] matter; (ii) the residual basis can be chosen so the position embedding is the literal "
               "one-hot of the position (first 6 coordinates) with two zero coordinates, which makes the embedding stage a pure "
               "one-hot token code with a 0/1 position offset and costs no constants; (iii) in that basis the whole network is "
               "h = tok_code + pos_onehot, a0 = softmax(h A0 h^T), r0 = h + sum_k a0[q,k] h_k VO0, a1 = softmax(r0 A1 r0^T), "
               "out = r0[5] G + sum_k a1[5,k] (r0_k H) with A0, A1, VO0, G, H fixed 8x8 / 8x3 tables. The tables are being "
               "measured; the readout below is still the placeholder rule (count k of positions p with token p = position p).")

def model(x, iv):
    x = np.asarray(x)
    E = np.eye(6)
    toks = [iv("tok%d" % p, E[x[:, p].astype(int)]) for p in range(6)]
    B = x.shape[0]
    k = np.zeros(B)
    for p in range(6):
        k = k + (toks[p][:, p]).astype(float)   # 1 when token at p equals p
    sc = np.stack([np.zeros(B), 0.4 - k, -0.6 - k, -1.6 - k], 1)
    return np.concatenate([sc, np.full((B, 2), -(10 ** 9))], 1)

ALIGN = {"tok%d" % p: ["embed"] for p in range(6)}
