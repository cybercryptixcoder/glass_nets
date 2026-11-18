import numpy as np

DESCRIPTION = (
    "The network reads a function f on positions 0..4. Layer-0 attention at query "
    "position i looks up the position j=f(i) (the token value written at i) and hence "
    "moves f(f(i)) into position i. The construction keeps the token code at the "
    "embedding stage, the layer-0 lookup pattern, and the layer-0 looked-up token. "
    "From these it reconstructs the represented function and classifies its longest "
    "cycle: following i -> f(i) until a value repeats gives each cycle length; the "
    "largest such length L is output as class L-1. The final class is represented as "
    "a one-hot code at the last attention stage."
)

def _tok4(vals):
    B, P = vals.shape
    out = np.zeros((B, P * 4))
    for p in range(P):
        for v in range(4):
            out[:, p * 4 + v] = (vals[:, p] == v)
    return out

def _dec4(code):
    a = code.reshape(-1, 5, 4)
    s = a.sum(axis=2)
    d = a.argmax(axis=2)
    return np.where(s > 0.5, d, 4).astype(int)

def model(x, iv):
    B = x.shape[0]
    P = 5
    V = 5
    ar = np.arange(B)

    # Embedding stage: each position's token, in a 4-coordinate affine token code.
    tok = iv("tok", _tok4(x))
    t = _dec4(tok)

    # Layer-0 attention: query i selects the key position j = f(i) = token at i.
    a0 = np.zeros((B, P * V))
    for i in range(P):
        a0[ar, i * V + t[:, i]] = 1.0
    a0 = iv("a0", a0)
    g = a0.reshape(-1, P, V).argmax(axis=2)

    # Layer-0 attention output token at position i: f(f(i)) = t[f(i)].
    o0tok = np.zeros_like(t)
    for i in range(P):
        o0tok[:, i] = t[ar, g[:, i]]
    o0 = iv("o0", _tok4(o0tok))

    # Reconstruct the represented function h from the first lookup g and the
    # layer-0 output tokens.  Normally g=f and h=f; under an intervention the
    # second-iterate values override consistently with the mixed state.
    o0v = _dec4(o0)
    h = g.copy()
    for i in range(P):
        h[ar, g[:, i]] = o0v[:, i]

    # Longest cycle: follow every start for at most 5 steps; first repeat gives
    # that start's cycle length.  Take the maximum over starts.
    seq = np.zeros((B, P, 6), dtype=int)
    seq[:, :, 0] = np.arange(P)
    for k in range(1, 6):
        seq[:, :, k] = h[ar[:, None], seq[:, :, k - 1]]
    cyc = np.zeros((B, P), dtype=int)
    for k in range(1, 6):
        for j in range(k):
            hit = (seq[:, :, k] == seq[:, :, j]) & (cyc == 0)
            cyc = np.where(hit, k - j, cyc)
    longest = cyc.max(axis=1)

    # Output class c means longest cycle length c+1.
    cls = np.zeros((B, V))
    cls[ar, longest - 1] = 1.0
    cls = iv("cls", cls)
    return 10.0 * cls

ALIGN = {
    "tok": ["embed"],
    "a0": ["attn0.pattern"],
    "o0": ["attn0.out"],
    "cls": ["attn1.out", "resid1.mid"],
}