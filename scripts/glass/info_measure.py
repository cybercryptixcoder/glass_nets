"""Rebuild step 1 (INTENT.md): does 'information versus the copy frontier' separate copies from explanations?

Pre-registered in docs/glass/DIARY.md (11:10, Nov 18) before this file was written. No AI calls.
B  = raw LZMA bits of a construction's code (DESCRIPTION, ALIGN, comments and docstrings removed by an ast round-trip).
F  = agreement with the network's decisions on the target's inputs.
C(F) = bits of the cheapest copy (weights quantised to k bits; uniform-integer or codebook form) with fidelity >= F.
S  = B / C(F).  Must-fail: S >= 0.8.  Must-pass: S <= 0.5.
"""
import ast, base64, json, lzma, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab"))
import judge, nets  # noqa: E402

FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def bits(code_text):
    return 8 * len(lzma.compress(code_text.encode(), format=lzma.FORMAT_RAW, filters=FILTERS))


def strip(source):
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("DESCRIPTION", "ALIGN") for t in node.targets):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue
        body.append(node)
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):   # drop docstrings inside functions
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


FWD = {
    "rnn_relu": """
def model(x, iv):
    s = np.zeros((len(x), W['W_h'].shape[0]))
    for t in range(x.shape[1]):
        s = np.maximum(x[:, t:t+1] * W['W_x'] + s @ W['W_h'], 0)
    return s @ W['W_o']
""",
    "transformer": """
def model(x, iv):
    h = W['W_E'][x] + W['W_pos'][:x.shape[1]]
    l = 0
    while f'L{l}.W_Q' in W:
        q, k, v = (np.einsum('bpd,hde->bhpe', h, W[f'L{l}.W_' + n]) for n in 'QKV')
        a = np.einsum('bhpe,bhqe->bhpq', q, k) / np.sqrt(q.shape[-1])
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        h = h + np.einsum('bhpq,bhqe,hed->bpd', a, v, W[f'L{l}.W_O'])
        l += 1
    return h[:, -1] @ W['W_U']
""",
}


def weight_keys(meta, W):
    return [k for k in W if k not in meta.get("protected", [])]


def quant_uniform(W, keys, k):
    Q, text = {}, ["import numpy as np", "W = {}"]
    levels = 2 ** (k - 1) - 1 if k > 1 else 1
    for key in keys:
        w = W[key]; s = np.abs(w).max() / levels if np.abs(w).max() > 0 else 1.0
        qi = np.round(w / s).astype(int)
        Q[key] = qi * s
        text.append(f"W[{key!r}] = np.array({qi.ravel().tolist()}).reshape({list(w.shape)}) * {float(s)!r}")
    return Q, "\n".join(text)


def quant_codebook(W, keys, k):
    allw = np.concatenate([W[key].ravel() for key in keys])
    n = min(2 ** k, len(np.unique(allw)))
    cents = np.quantile(allw, (np.arange(n) + 0.5) / n)
    for _ in range(30):  # 1-D k-means
        idx = np.abs(allw[:, None] - cents[None]).argmin(1)
        for c in range(n):
            if (idx == c).any():
                cents[c] = allw[idx == c].mean()
    cents = np.round(cents, 4)
    Q, text, off = {}, ["import numpy as np", f"V = np.array({cents.tolist()})", "W = {}"], 0
    for key in keys:
        w = W[key]; ii = np.abs(w.ravel()[:, None] - cents[None]).argmin(1)
        Q[key] = cents[ii].reshape(w.shape)
        text.append(f"W[{key!r}] = V[{ii.tolist()}].reshape({list(w.shape)})")
    return Q, "\n".join(text)


FWD_GAUGE = {
    "rnn_relu": FWD["rnn_relu"],
    "transformer": """
def model(x, iv):
    h = W['W_E'][x] + W['W_pos'][:x.shape[1]]
    l = 0
    while f'QK{l}' in W:
        a = np.einsum('bpd,hde,bqe->bhpq', h, W[f'QK{l}'], h) / np.sqrt(W['k'])
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        h = h + np.einsum('bhpq,bqd,hde->bpe', a, h, W[f'OV{l}'])
        l += 1
    return h[:, -1] @ W['W_U']
""",
}


def gauge(meta, W):
    """The network in a reduced parametrisation (T2): transformers as QK = W_Q W_K^T and OV = W_V W_O products;
    ReLU RNNs with every hidden unit rescaled so that |W_x| = 1 (ReLU commutes with positive scales).
    Returns (weights for nets.forward-equivalent evaluation, the reduced tensors to quantise, kind)."""
    if meta["forward"] == "transformer":
        R = {k: W[k] for k in ("W_E", "W_pos", "W_U")}
        l = 0
        while f"L{l}.W_Q" in W:
            R[f"QK{l}"] = np.einsum("hde,hfe->hdf", W[f"L{l}.W_Q"], W[f"L{l}.W_K"])
            R[f"OV{l}"] = np.einsum("hde,hef->hdf", W[f"L{l}.W_V"], W[f"L{l}.W_O"])
            l += 1
        return R
    if meta["forward"] == "rnn_relu":
        s = np.abs(W["W_x"]).ravel(); s = np.where(s > 1e-12, s, 1.0)
        return {"W_x": W["W_x"] / s[None], "W_h": W["W_h"] * s[:, None] / s[None], "W_o": W["W_o"] * s[:, None]}
    return None


def run_gauge(meta, R, X, k):
    if meta["forward"] == "rnn_relu":
        return nets.forward("rnn_relu", R, X)
    h = R["W_E"][X] + R["W_pos"][:X.shape[1]]; l = 0
    while f"QK{l}" in R:
        a = np.einsum("bpd,hde,bqe->bhpq", h, R[f"QK{l}"], h) / np.sqrt(k)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        h = h + np.einsum("bhpq,bqd,hde->bpe", a, h, R[f"OV{l}"]); l += 1
    return h[:, -1] @ R["W_U"]


def fidelity(meta, W, X, out_candidate):
    net = nets.forward(meta["forward"], W, X)
    return float(judge.same(net, out_candidate).mean())


def frontier(meta, W, X):
    keys = weight_keys(meta, W); pts = []
    for k in range(1, 17):
        for form, fn in (("uniform", quant_uniform), ("codebook", quant_codebook)):
            if form == "codebook" and k > 10:
                continue
            Q, text = fn(W, keys, k)
            WQ = dict(W); WQ.update(Q)
            f = fidelity(meta, W, X, nets.forward(meta["forward"], WQ, X))
            pts.append((f, bits(text + FWD[meta["forward"]]), f"{form}{k}"))
    R = gauge(meta, W)
    if R is not None:
        kdim = W["L0.W_Q"].shape[-1] if meta["forward"] == "transformer" else 0
        rk = list(R)
        for k in range(1, 17):
            for form, fn in (("g-uniform", quant_uniform), ("g-codebook", quant_codebook)):
                if form == "g-codebook" and k > 10:
                    continue
                Q, text = fn(R, rk, k)
                f = fidelity(meta, W, X, run_gauge(meta, Q, X, kdim))
                pts.append((f, bits(text + (f"\nW['k'] = {kdim}" if kdim else "") + FWD_GAUGE[meta["forward"]]), f"{form}{k}"))
    return pts


def C_of(pts, F):
    ok = [b for f, b, _ in pts if f >= F - 1e-12]
    return min(ok) if ok else max(b for _, b, _ in pts)


def run_source(source, X):
    model, _align = judge.load_construction(source)
    return model(X, lambda name, v: v)


def disguises(meta, W):
    """Written before the measure was coded (pre-registration D1-D6): exact copies in new forms."""
    keys = weight_keys(meta, W); fwd = FWD[meta["forward"]]; out = {}
    def wrap(lines): return "import numpy as np, base64, zlib\nW = {}\n" + "\n".join(lines) + fwd
    out["D1 float32 base64"] = wrap([f"W[{k!r}] = np.frombuffer(base64.b64decode({base64.b64encode(W[k].astype(np.float32).tobytes()).decode()!r}), np.float32).astype(float).reshape({list(W[k].shape)})" for k in keys])
    out["D2 decimals 6sf"] = wrap([f"W[{k!r}] = np.array({[float(f'{v:.6g}') for v in W[k].ravel()]}).reshape({list(W[k].shape)})" for k in keys])
    out["D3 int16 over scale"] = wrap([f"W[{k!r}] = np.array({np.round(W[k].ravel() / (np.abs(W[k]).max() / 32767)).astype(int).tolist()}).reshape({list(W[k].shape)}) * {float(np.abs(W[k]).max() / 32767)!r}" for k in keys])
    allw = np.concatenate([W[k].ravel() for k in keys]); cb = np.unique(np.round(allw, 3))[:256] if len(np.unique(np.round(allw, 3))) <= 256 else None
    if cb is None:
        cb = np.round(np.quantile(allw, (np.arange(256) + 0.5) / 256), 4)
    lines, = [[f"V = np.array({cb.tolist()})"] + [f"W[{k!r}] = V[{np.abs(W[k].ravel()[:, None] - cb[None]).argmin(1).tolist()}].reshape({list(W[k].shape)})" for k in keys]]
    out["D4 codebook256"] = wrap(lines)
    lines = []
    for k in keys:
        v = W[k].ravel(); p = np.argsort(v)
        lines.append(f"_s = np.array({[round(float(x), 4) for x in v[p]]}); _p = np.array({p.tolist()}); _w = np.empty(len(_p)); _w[_p] = _s; W[{k!r}] = _w.reshape({list(W[k].shape)})")
    out["D5 sorted+perm"] = wrap(lines)
    out["D6 zlib float16 base64"] = wrap([f"W[{k!r}] = np.frombuffer(zlib.decompress(base64.b64decode({base64.b64encode(zlib.compress(W[k].astype(np.float16).tobytes(), 9)).decode()!r})), np.float16).astype(float).reshape({list(W[k].shape)})" for k in keys])
    return out


def construction_file(target, run_id):
    return (ROOT / "lab" / "constructions" / target / run_id / "construction.py").read_text()


CASES = [  # (label, target, run_id, variant, expected)
    ("M2,2 prior-only account (r1)", "zoo_2nd_argmax_2_2", "20261118-015915-a7c8", "original", "pass"),
    ("M4,3 account c928 (r1)", "zoo_2nd_argmax_4_3", "20261118-015037-c928", "original", "pass"),
    ("lc_8_6 census rule (r2 b)", "lc_8_6", "20261118-055610-d7df", "original", "pass"),
    ("lc_3_4 maina int copy", "lc_3_4", "20261118-045127-234b", "original", "fail"),
    ("lc_4_5 mainb int copy", "lc_4_5", "20261118-050059-93e7", "original", "fail"),
    ("lc_6_5 maina int copy", "lc_6_5", "20261118-050628-dba7", "original", "fail"),
    ("M8,5 v1a int copy", "zoo_2nd_argmax_8_5", "20261118-053747-35b7", "original", "fail"),
    ("M16,10 v1a int copy", "zoo_2nd_argmax_16_10", "20261118-062600-878d", "original", "fail"),
    ("lc_4_5 maina near-copy", "lc_4_5", "20261118-045956-6821", "original", "fail"),
    ("lc_8_6 maina near-copy", "lc_8_6", "20261118-055556-e50d", "original", "fail"),
    ("M4,4 v1a near-copy", "zoo_2nd_argmax_4_4", "20261118-051642-3aea", "original", "fail"),
    ("M8,5 v2a near-copy", "zoo_2nd_argmax_8_5", "20261118-054317-293b", "original", "fail"),
    ("M4,3 v2a (0.78, unlabelled)", "zoo_2nd_argmax_4_3", "20261118-045809-3312", "original", "report"),
]


def queue_case(name, label, expected, variant="original"):
    q = json.loads((ROOT / "results/glass/round2/queue.json").read_text())[name]
    return (label, q["target"], q["run_id"], variant, expected)


def main():
    cases = list(CASES)
    for name, lab_ in [("zoo_2nd_argmax_8_8__v1b", "M8,8 v1b int copy"), ("zoo_2nd_argmax_16_10__v2a", "M16,10 v2a codebook copy"),
                       ("zoo_2nd_argmax_16_10__v1b", "M16,10 v1b near-copy")]:
        cases.append(queue_case(name, lab_, "fail"))
    cases.append(queue_case("lc_3_4__shuffled", "lc_3_4 SHUFFLED-net account", "fail", "shuffled"))
    cache, rows = {}, []
    for label, tgt, rid, var, exp in cases:
        if (tgt, var) not in cache:
            meta, W, _ = judge.load_target(tgt, var, 0)
            X = np.load(ROOT / "lab/models" / tgt / "inputs.npy")
            cache[(tgt, var)] = (meta, W, X, frontier(meta, W, X))
        meta, W, X, pts = cache[(tgt, var)]
        src = construction_file(tgt, rid)
        try:
            F = fidelity(meta, W, X, run_source(src, X))
        except Exception as e:  # noqa: BLE001
            rows.append((label, exp, None, None, None, None, f"error: {e}")); continue
        B = bits(strip(src)); C = C_of(pts, F)
        rows.append((label, exp, F, B, C, B / C, ""))
    for tgt in ("zoo_2nd_argmax_8_5", "lc_6_5", "zoo_2nd_argmax_16_10"):
        if (tgt, "original") not in cache:
            meta, W, _ = judge.load_target(tgt, "original", 0)
            X = np.load(ROOT / "lab/models" / tgt / "inputs.npy")
            cache[(tgt, "original")] = (meta, W, X, frontier(meta, W, X))
        meta, W, X, pts = cache[(tgt, "original")]
        for dname, text in disguises(meta, W).items():
            env = {"__name__": "__disguise__"}
            exec(text, env)
            F = fidelity(meta, W, X, env["model"](X, None))
            B = bits(strip(text)); C = C_of(pts, F)
            rows.append((f"{tgt} {dname}", "fail", F, B, C, B / C, ""))
    print("| case | expected | fidelity F | bits B | copy bits C(F) | S = B/C | ok? |")
    print("|---|---|---|---|---|---|---|")
    allok = True
    for label, exp, F, B, C, S, err in rows:
        if S is None:
            print(f"| {label} | {exp} | - | - | - | - | {err} |"); allok = False; continue
        ok = (S >= 0.8) if exp == "fail" else (S <= 0.5) if exp == "pass" else None
        allok &= ok is not False
        print(f"| {label} | {exp} | {F:.3f} | {B} | {C} | {S:.2f} | {'-' if ok is None else ('yes' if ok else 'NO')} |")
    print("\nFrontiers (fidelity: bits):")
    for (tgt, var), (_, _, _, pts) in cache.items():
        best = {}
        for f, b, nm in pts:
            for thr in (0.9, 0.95, 0.99, 1.0):
                if f >= thr and b < best.get(thr, (1e18,))[0]:
                    best[thr] = (b, nm)
        print(f"- {tgt} [{var}]: " + "; ".join(f">={t:.2f}: {v[0]} bits ({v[1]})" for t, v in sorted(best.items())))
    print("\nPRE-REGISTERED CRITERION MET:", allok)


if __name__ == "__main__":
    main()
