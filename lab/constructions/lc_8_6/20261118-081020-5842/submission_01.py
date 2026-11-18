import numpy as np

DESCRIPTION = (
    "The network is a two-layer pointer-chasing transformer on a function f. "
    "At layer 0 each query position i attends to the key position equal to f(i), "
    "so the attention output carries the token f(f(i)) = f^2(i). At layer 1 each "
    "query i attends to the position equal to f^2(i); the value there contains "
    "both the original token f(j)=f^3(i) and the layer-0 output f^2(j)=f^4(i). "
    "The residual stream therefore carries the iterates f, f^2, f^3, f^4 at each "
    "position, and the final readout classifies the longest cycle from the function."
)

def _onehot(vals):
    # vals: [B, 6] integer tokens; returns [B, 6, 6]
    return np.eye(6, dtype=float)[vals]

def model(x, iv):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    B = x.shape[0]
    f1_raw = x.astype(np.int64)

    # Embedding stage: one-hot token at every position.
    f1_oh = iv("f1_tok", _onehot(f1_raw).reshape(B, 36))
    f1 = np.argmax(f1_oh.reshape(B, 6, 6), axis=2)

    # Layer 0: query i -> key position f(i), value f(f(i)) = f^2(i).
    f2_raw = np.take_along_axis(f1, f1, axis=1)
    f2_oh = iv("f2_tok", _onehot(f2_raw).reshape(B, 36))
    f2 = np.argmax(f2_oh.reshape(B, 6, 6), axis=2)

    ar = np.arange(6)
    pat0 = (ar[None, None, :] == f1[:, :, None]).astype(float).reshape(B, 36)
    pat0 = iv("pattern0", pat0)

    # Layer 1: query i -> key position f^2(i), value contains f^3(i), f^4(i).
    pat1 = (ar[None, None, :] == f2[:, :, None]).astype(float).reshape(B, 36)
    pat1 = iv("pattern1", pat1)

    f3 = np.take_along_axis(f1, f2, axis=1)
    f4 = np.take_along_axis(f2, f2, axis=1)
    f3_oh = iv("f3_tok", _onehot(f3).reshape(B, 36))
    f4_oh = iv("f4_tok", _onehot(f4).reshape(B, 36))

    # Exact task readout: longest cycle length minus one.
    out = np.zeros((B, 6), dtype=float)
    for b in range(B):
        f = f1[b]
        best = 0
        for start in range(6):
            seen = {}
            cur = int(start)
            t = 0
            while cur not in seen:
                seen[cur] = t
                cur = int(f[cur])
                t += 1
            L = t - seen[cur]
            if L > best:
                best = L
        out[b, best - 1] = 10.0
    return out

ALIGN = {
    "f1_tok": ["embed"],
    "pattern0": ["attn0.pattern"],
    "f2_tok": ["attn0.out"],
    "pattern1": ["attn1.pattern"],
    "f3_tok": ["attn1.out"],
    "f4_tok": ["attn1.out"],
}