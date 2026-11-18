
import numpy as np

DESCRIPTION = ("Stage-by-stage re-implementation in progress. Established and used here: (i) the decision is the argmax of "
               "position 5's residual times W_U, and that argmax is always one of classes 0..3 (classes 4 and 5 are never the "
               "decision), so only the three class differences W_U[:,1:4]-W_U[:,0:1] matter; (ii) there is a residual basis in "
               "which the position embedding is the literal one-hot of the position in the first six coordinates and zero in the "
               "last two, so the embedding stage is exactly a one-hot token code plus a 0/1 position offset and costs no "
               "constants (verified: explained 1.000 at site embed); (iii) in that basis the whole network is "
               "h = tok + pos, a0 = softmax(h A0 h^T), r0 = h + sum_k a0[q,k] (h_k VO0), a1 = softmax(r0 A1 r0^T), "
               "scores = r0[5] G + sum_k a1[5,k] (r0_k H), with five measured 8x8/8x3 tables A0, A1, VO0, G, H only. "
               "The tables are not yet written into this file, so the readout below is still the provisional rule "
               "(count k of positions p whose token equals p).")

def model(x, iv):
    x = np.asarray(x)
    E = np.eye(6)
    tok = [iv("tok%d" % p, E[x[:, p].astype(int)]) for p in range(6)]
    B = x.shape[0]
    k = np.zeros(B)
    for p in range(6):
        k = k + tok[p][:, p]
    sc = np.stack([np.zeros(B), 0.4 - k, -0.6 - k, -1.6 - k], 1)
    return np.concatenate([sc, np.full((B, 2), -(10 ** 9))], 1)

ALIGN = {"tok%d" % p: ["embed"] for p in range(6)}
