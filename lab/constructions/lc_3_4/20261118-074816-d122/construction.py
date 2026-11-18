import numpy as np

DESCRIPTION = (
    "Computes the longest cycle length of the input function f on {0,1,2,3}. "
    "The embedding site is exactly one-hot token indicators per position (an affine "
    "function of these spans the token+position embeddings).  The final residual at "
    "the last position holds a 3D code for the answer class: classes 0,1,2,3 map to "
    "(1,0,0),(0,1,0),(0,0,1),(0,0,0).  A linear barycentric readout recovers the "
    "four logits.  The model uses the iv-returned token and class variables so that "
    "embedding or final-residual interventions propagate causally."
)

CODE = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]])

def _lens(tt):
    B = tt.shape[0]
    L = np.zeros(B, dtype=int)
    for b in range(B):
        f = tt[b]
        best = 1
        for i in range(4):
            seen = []
            cur = int(i)
            while cur not in seen:
                seen.append(cur)
                cur = int(f[cur])
            cyc = len(seen) - seen.index(cur)
            if cyc > best:
                best = cyc
        L[b] = best - 1
    return L

def model(x, iv):
    x = np.asarray(x, dtype=int)
    B = x.shape[0]

    # one-hot token variable per position at the embedding site
    oh = []
    for p in range(4):
        v = np.zeros((B, 4))
        v[np.arange(B), x[:, p]] = 1.0
        oh.append(v)
    v0 = iv("tok0", oh[0])
    v1 = iv("tok1", oh[1])
    v2 = iv("tok2", oh[2])
    v3 = iv("tok3", oh[3])

    # decode the (possibly intervened) tokens
    tt = np.stack([np.argmax(v, axis=1) for v in (v0, v1, v2, v3)], axis=1)
    L = _lens(tt)

    # final-residual class code at the last position
    c = iv("cls", CODE[L])

    # barycentric readout: [x, y, z, 1-x-y-z]
    logits = np.stack([c[:, 0], c[:, 1], c[:, 2],
                       1.0 - c[:, 0] - c[:, 1] - c[:, 2]], axis=1)
    return logits

ALIGN = {
    "tok0": ["embed"],
    "tok1": ["embed"],
    "tok2": ["embed"],
    "tok3": ["embed"],
    "cls":  ["resid1.mid"],
}
