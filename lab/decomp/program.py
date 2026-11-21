"""D1 stage 2: from recovered input coordinates to a runnable program, and its verification.

Weights and internal activations only. The network's outputs (logits / argmax) are never used to fit anything: they
are used only by `verify_*` to score the program afterwards.

  internals(W, toks)            one-layer transformer forward pass returning every intermediate
  mlp_characters(...)           the MLP's output as a sum over characters of G x G (2-D Fourier in recovered coordinates)
  decode_outputs(...)           output classes -> group elements, read from W_U along the MLP's write directions
  make_program(...)             the program: c = OUT^-1( g(a) + g(b) ), with the tables printed
  verify_replacement(...)       M1
  verify_causal(...)            M2: patch only the recovered coordinates of one input inside the real network
"""
import itertools
import numpy as np


def internals(W, toks, patch=None):
    """Forward pass of lab's onelayer_transformer (no LayerNorm, no biases). patch(x0) may edit the input residual."""
    x = W["W_E"][toks] + W["W_pos"][None]
    if patch is not None:
        x = patch(x)
    q = np.einsum("bpd,hde->bhpe", x, W["W_Q"]); k = np.einsum("bpd,hde->bhpe", x, W["W_K"]); v = np.einsum("bpd,hde->bhpe", x, W["W_V"])
    att = np.einsum("bhpe,bhqe->bhpq", q, k) / np.sqrt(W["W_Q"].shape[-1])
    P = x.shape[1]
    att = np.where(np.triu(np.ones((P, P), bool), 1)[None, None], -1e9, att)
    att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
    z = np.einsum("bhpq,bhqe->bhpe", att, v)
    ao = np.einsum("bhpe,hed->bpd", z, W["W_O"])
    x1 = x + ao
    pre = x1[:, -1] @ W["W_in"]; post = np.maximum(pre, 0)
    mlp = post @ W["W_out"]
    x2 = x1[:, -1] + mlp
    return {"x0": x, "att": att, "x1": x1, "pre": pre, "post": post, "mlp": mlp, "x2": x2, "logits": x2 @ W["W_U"]}


def all_pairs(n):
    a = np.arange(n)
    return np.stack([np.repeat(a, n), np.tile(a, n), np.full(n * n, n)], 1)


def batched(W, toks, bs=4096, keys=("mlp", "x2", "logits"), patch=None):
    out = {k: [] for k in keys}
    for i in range(0, len(toks), bs):
        r = internals(W, toks[i:i + bs], patch=None if patch is None else (lambda x, i=i: patch(x, i)))
        for k in keys:
            out[k].append(r[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def group_elements(d):
    return np.array(list(itertools.product(*[range(x) for x in d]))) if d else np.zeros((1, 0), int)


def element_index(g, d):
    idx = np.zeros(len(g), int)
    for k, x in enumerate(d):
        idx = idx * x + g[:, k]
    return idx


def characters(d):
    """chi[u, h] = exp(2 pi i <u, h>/d) for u, h in G = Z_d1 x ... (|G| x |G|)."""
    els = group_elements(d); dd = np.array(d)
    return np.exp(2j * np.pi * ((els[:, None, :] * els[None, :, :]) / dd).sum(-1)), els


def mlp_characters(mlp, ga, gb, d, on_a=None, on_b=None):
    """MLP output M(a, b) (n x n x dm) re-indexed by the recovered elements of a and b, then 2-D Fourier over G x G.
    Returns power[u, v] (share of the MLP output's variance carried by chi_u(g_a) chi_v(g_b)) and the coefficient
    vectors Mhat[u, v] (complex, dm)."""
    n = int(np.sqrt(len(mlp)))
    M = mlp.reshape(n, n, -1)
    order = int(np.prod(d))
    ia, ib = element_index(ga, d), element_index(gb, d)
    keep_a = np.ones(n, bool) if on_a is None else on_a
    keep_b = np.ones(n, bool) if on_b is None else on_b
    # average symbols that share an element (2-to-1 maps); elements nobody maps to stay zero
    S = np.zeros((order, order, M.shape[-1])); C = np.zeros((order, order, 1))
    A, B = np.meshgrid(np.where(keep_a)[0], np.where(keep_b)[0], indexing="ij"); A, B = A.ravel(), B.ravel()
    np.add.at(S, (ia[A], ib[B]), M[A, B]); np.add.at(C, (ia[A], ib[B]), 1)
    S = S / np.maximum(C, 1)
    S = S - S.mean((0, 1), keepdims=True)
    chi, els = characters(d)
    T = np.einsum("ua,abd->ubd", chi.conj(), S)
    Mhat = np.einsum("vb,ubd->uvd", chi.conj(), T) / order ** 2
    power = (np.abs(Mhat) ** 2).sum(-1)
    return power / power.sum(), Mhat, els


def diagonal_share(power, d):
    """Share of MLP power on characters of the form chi_u(g_a) chi_u(g_b) = chi_u(g_a + g_b): 1.0 means the MLP output
    depends on a and b only through g_a + g_b."""
    return float(np.trace(power))


def decode_outputs(W_U, Mhat, power, d, top=16, assign=True):
    """Each output class c gets the element h maximising sum_u Re( conj(chi_u(h)) * <W_U[:, c], Mhat[u, u]> ) over the
    strongest diagonal characters u: the class whose readout direction lines up with what the MLP writes for h."""
    diag = np.real(np.diag(power))
    us = np.argsort(diag)[::-1][:top]
    chi, els = characters(d)
    score = np.zeros((W_U.shape[1], len(els)))
    for u in us:
        r = W_U.T @ Mhat[u, u]                      # (n_out,) complex: class c's projection on what the MLP writes for chi_u
        score += np.real(r[:, None] * chi[u][None, :])
    h = score.argmax(1)
    srt = np.sort(score, 1)
    n_argmax_distinct = len(set(h.tolist()))
    leftover = []
    if assign and W_U.shape[1] >= len(els):
        # one class per element (an assignment, not a per-class argmax); reported: how many classes it moved. With more
        # classes than elements, the classes no element claims are returned as `leftover`
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-score)
        h2 = np.full_like(h, -1); h2[r] = c
        leftover = [int(x) for x in np.where(h2 < 0)[0]]
        moved = int(((h2 != h) & (h2 >= 0)).sum()); h = np.where(h2 >= 0, h2, h)
    else:
        moved = 0
    return {"h": els[h], "h_index": h, "leftover": leftover, "margin": (srt[:, -1] - srt[:, -2]) / np.abs(srt[:, -1]).clip(1e-12), "chars": us.tolist(),
            "argmax_distinct": n_argmax_distinct, "moved_by_assignment": moved}


def make_program(ga, h_out, d, on_a=None, leftover=None):
    """Return a Python function f(a, b) and its source text. ga: element of each input symbol; h_out: element of each
    output class. Symbols off every circle (on_a False) have no element: the program reports them as unexplained."""
    n = len(ga); dd = np.array(d)
    leftover = leftover or []
    inv = {}
    for c, h in enumerate(h_out):
        if c in leftover:
            continue
        inv.setdefault(tuple(int(x) for x in h), c)
    G_in = [tuple(int(x) for x in g) for g in ga]
    on = np.ones(n, bool) if on_a is None else on_a
    off = [int(i) for i in np.where(~on)[0]]
    absorbing = leftover[0] if (len(off) == 1 and len(leftover) == 1) else None

    def f(a, b):
        if not (on[a] and on[b]):
            return absorbing
        s = tuple(int(x) for x in (np.array(G_in[a]) + np.array(G_in[b])) % dd)
        return inv.get(s)
    src = [f"# Decompiled program (group Z_{' x Z_'.join(map(str, d))}); tables read from the weights",
           f"G = {list(d)}",
           f"ELEMENT = {{  # input symbol -> group element\n    " + ", ".join(f"{i}: {G_in[i]}" for i in range(n) if on[i]) + "\n}",
           f"CLASS_OF = {{  # group element -> output class\n    " + ", ".join(f"{k}: {v}" for k, v in sorted(inv.items())) + "\n}",
           *( [f"ABSORBING = {absorbing}  # symbol {off[0]} sits off every circle and output class {absorbing} is claimed by no element"] if absorbing is not None else []),
           "def f(a, b):",
           *( ["    if a not in ELEMENT or b not in ELEMENT:", "        return ABSORBING"] if absorbing is not None else []),
           "    s = tuple((x + y) % m for x, y, m in zip(ELEMENT[a], ELEMENT[b], G))",
           "    return CLASS_OF[s]"]
    return f, "\n".join(src)


def verify_replacement(f, net_pred, n):
    toks = all_pairs(n)
    prog = np.array([(-1 if (r := f(int(a), int(b))) is None else r) for a, b, _ in toks])
    defined = prog >= 0
    return {"agreement": float((prog == net_pred).mean()), "defined": float(defined.mean()),
            "agreement_where_defined": float((prog[defined] == net_pred[defined]).mean()) if defined.any() else 0.0}


def verify_causal(W, planes, ga, f, n, n_patches=2000, seed=0, center=None, random_planes=False):
    """M2. For input (a, b), replace a's coordinates in the recovered planes (projection of its embedding onto the
    planes' span) by those of a', leaving everything else in a's embedding untouched. If the planes carry what the
    network uses, the network now answers f(a', b)."""
    rng = np.random.default_rng(seed)
    E = W["W_E"]
    dm = E.shape[1]
    if random_planes:
        Q = np.linalg.qr(rng.standard_normal((dm, 2 * len(planes))))[0]
    else:
        Q = np.linalg.qr(np.concatenate([pl.get("Qo", pl["Q"]) for pl in planes], 1))[0]
    Pj = Q @ Q.T
    A = rng.integers(0, n, n_patches); A2 = rng.integers(0, n, n_patches); B = rng.integers(0, n, n_patches)
    toks = np.stack([A, B, np.full(n_patches, n)], 1)
    delta = (E[A2] - E[A]) @ Pj                                      # move a onto a' inside the planes only
    def patch(x, i0):
        x = x.copy(); x[:, 0] += delta[i0:i0 + len(x)]; return x
    out = batched(W, toks, keys=("logits",), patch=patch)["logits"].argmax(-1)
    want = np.array([(-1 if (r := f(int(a2), int(b))) is None else r) for a2, b in zip(A2, B)])
    ok = want >= 0
    changed = A2 != A
    return {"success": float((out[ok & changed] == want[ok & changed]).mean()), "n": int((ok & changed).sum())}
