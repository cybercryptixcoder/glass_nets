"""The box on the bench's S0 networks (`bench/accept/INTERFACE.md`, interface bench-accept/v1): Tracr-compiled
transformers whose residual stream is in an arbitrary orthonormal basis, with no labels published.

The box is given the task folder only (weights, sites, the inputs the behaviour is explained on). Steps:
1. Recover a frame in which each variable occupies its own directions: from the weights, the token directions and the
   constant direction (every token's embedding is its own direction plus one shared direction, which is the part of
   their span orthogonal to all their differences) and the position directions; then, sublayer by sublayer, the
   subspace it writes (the span of its change to the residual stream on the inputs). In a written subspace, a variable
   whose activity takes exactly as many distinct nonzero points as the subspace has dimensions, and those points are
   orthonormal, is categorical, and the points are its directions; a written subspace of one dimension is numerical,
   signed so that its values are mostly non-negative. Anything else is kept as its principal directions.
2. Run the same box as on the public programs (`box/tracr_box.py`: map, define, invent, assemble), in that frame.
3. Write the blueprint in the bench's format, each location turned back into the network's own basis.
Run: python3 -m box.bench_s0 <task folder> <out folder>   (or --draw <draw folder> <out folder>)"""
import json
import os
import sys

import numpy as np

BENCH = os.environ.get("BENCH_DIR", "/home/claude/bench_v2_wt/bench/accept")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BENCH not in sys.path:
    sys.path.append(BENCH)
import netlib  # noqa: E402  (the bench's loader: the box may use it, INTERFACE.md §1)

from box import tok_fast as F  # noqa: E402
from box import tracr_box as TB  # noqa: E402


def _orth(M, tol=1e-9):
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > tol * max(s[0], 1e-12)]


def recover_frame(bnet, seqs, tol=1e-6):
    """Returns (F [d, d] orthonormal, names, kinds, error). Every token's embedding is its own direction plus the
    shared constant direction ("one"). The constant direction is not in the span of the embeddings: it is the
    smallest-norm vector with unit inner product with every embedding, plus the one direction the network uses outside
    everything else, scaled to unit length; of its two signs, the one under which the network's input weights are
    sparsest in the frame (Tracr writes and reads whole directions)."""
    P = bnet.P
    tokE, posE = P["token_embed"], P["pos_embed"]
    d = tokE.shape[1]
    inv = {v: k for k, v in bnet.tok_id.items()}
    nz = [t for t in range(len(tokE)) if np.linalg.norm(tokE[t]) > 1e-9]
    R = tokE[nz]
    pos = [(p_, posE[p_] / np.linalg.norm(posE[p_])) for p_ in range(1, len(posE)) if np.linalg.norm(posE[p_]) > 1e-9]
    base = _orth(np.column_stack([R.T] + [e[:, None] for _, e in pos]))
    res = bnet.run({"tokens": seqs}, return_sites=True)
    S = res["sites"]
    _, mask = bnet._encode(seqs)
    order = sorted(S, key=lambda n: int(n.split("_")[1]))
    written, wnames, kinds = [], [], {}
    for k in range(1, len(order)):
        H = S[order[k]][mask]
        D = H - S[order[k - 1]][mask]
        Q = np.column_stack([base] + [w[:, None] for w in written]) if written else base
        D = D - (D @ Q) @ Q.T                       # what is new at this sublayer
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        r = int((sv > tol * max(sv[0], 1e-12)).sum()) if sv.size and sv[0] > 1e-9 else 0
        if r == 0:
            continue
        W = Vt[:r]
        X = H @ W.T
        pts = np.unique(np.round(X, 5), axis=0)
        pts = pts[np.linalg.norm(pts, axis=1) > 1e-4]
        if r >= 2 and len(pts) == r and np.allclose(pts @ pts.T, np.eye(r), atol=1e-4):
            dirs = [p_ @ W for p_ in pts]; kind = "categorical"
        elif r == 1:
            x = X[:, 0]
            sgn = 1.0 if (x.mean() > 1e-9 or (abs(x.mean()) <= 1e-9 and x[np.argmax(np.abs(x))] >= 0)) else -1.0
            dirs = [sgn * W[0]]; kind = "numerical"
        else:
            dirs = list(W); kind = "mixed"
        for i, e in enumerate(dirs):
            written.append(e / np.linalg.norm(e)); wnames.append(f"w{k}_{i}")
        kinds[k] = (kind, r)
    used = np.column_stack([base] + [w[:, None] for w in written]) if written else base
    Nc = _orth(np.eye(d) - used @ np.linalg.pinv(used), 1e-6)     # what nothing above spans
    e_min = R.T @ np.linalg.solve(R @ R.T, np.ones(len(R)))
    c = float(np.sqrt(max(0.0, 1.0 - e_min @ e_min)))
    reads = []
    for l in range(bnet.n_layers):
        reads += [P[f"layer_{l}/attn/{n}/w"] for n in ("query", "key", "value")] + [P[f"layer_{l}/mlp/linear_1/w"]]

    def build(one):
        cols = [one] + [R[i] - one for i in range(len(R))] + [e for _, e in pos] + written
        names = ["one"] + [f"tokens:{inv[t]}" for t in nz] + [f"indices:{p_ - 1}" for p_, _ in pos] + wnames
        Q = np.column_stack(cols)
        if Q.shape[1] < d:
            rest = _orth(np.eye(d) - Q @ np.linalg.pinv(Q), 1e-6)[:, :d - Q.shape[1]]
            Q = np.column_stack([Q, rest]); names += [f"unused_{i}" for i in range(rest.shape[1])]
        return Q, names
    options = []
    if c > 1e-6 and Nc.shape[1] >= 1:
        for sgn in (1.0, -1.0):
            options.append(e_min + sgn * c * Nc[:, 0])
    else:
        options.append(e_min / np.linalg.norm(e_min))
    best = None
    for one in options:
        Q, names = build(one)
        nnz = sum(int((np.abs(Q.T @ W) > 1e-6).sum()) for W in reads)
        err = float(np.abs(Q.T @ Q - np.eye(Q.shape[1])).max()) if Q.shape[1] == d else 1.0
        if best is None or (err, nnz) < (best[3], best[4]):
            best = (Q, names, kinds, err, nnz)
    return best[0], best[1], best[2], best[3]


class FrameNet:
    """The bench network seen in the recovered frame: sites hold h @ F; edits written in the frame are mapped back."""

    def __init__(self, bnet, Fm):
        self.b, self.F = bnet, Fm
        self.d = Fm.shape[1]
        self.sites = list(bnet.site_names)
        self.n_layers = bnet.n_layers
        self.labels = []
        self.numerical_output = bnet.meta.get("readout_kind") == "numerical"

    def _wrap(self, edits):
        if not edits:
            return None
        Fm = self.F
        return {s: (lambda fn: (lambda h: fn(h @ Fm) @ Fm.T))(fn) for s, fn in edits.items()}

    def run(self, seqs, edits=None, return_sites=False):
        res = self.b.run({"tokens": seqs}, edits=self._wrap(edits), return_sites=True)
        sites = {s: h @ self.F for s, h in res["sites"].items()}
        _, mask = self.b._encode(seqs)
        h = sites[self.sites[-1]]
        return (h, sites, mask) if return_sites else h

    def outputs(self, seqs, edits=None):
        return self.b.run({"tokens": seqs}, edits=self._wrap(edits))["outputs"]

    def reads(self, k):
        layer, is_mlp = (k - 1) // 2, (k - 1) % 2 == 1
        p = f"layer_{layer}"
        mats = [self.b.P[f"{p}/mlp/linear_1/w"]] if is_mlp else \
            [self.b.P[f"{p}/attn/{n}/w"] for n in ("query", "key", "value")]
        rows = np.zeros(self.d, bool)
        for W in mats:
            rows |= np.abs(self.F.T @ W).max(1) > 1e-6
        return set(np.nonzero(rows)[0].tolist())


def bench_source(name, src):
    """The bench's vocabulary passes an invented operation the whole SOp (a list of positions per input), not one
    value; the box's operations are written for one value. Same formula, applied position by position, numpy
    arithmetic (division by zero gives inf, as in the box's own evaluation); `is` is not allowed there, so `== None`. DIARY 15:30."""
    import re
    expr = src.strip().splitlines()[-1].strip()
    assert expr.startswith("return "), src
    body = re.sub(r"\bx\b", "np.float64(v)", expr[len("return "):])
    return (f"def {name}(x):\n    return [[None if v == None else {body} for v in row] for row in x]\n")


def explain(task_dir, out_dir, verbose=True):
    task = json.load(open(os.path.join(task_dir, "task.json")))
    bnet = netlib.load_network(task_dir, task)
    seqs = json.load(open(os.path.join(task_dir, task["inputs"]["path"])))["tokens"]
    vocab_tokens = sorted({t for s in seqs for t in s}, key=lambda v: (isinstance(v, str), v))
    max_len = int(bnet.meta["max_seq_len"])
    Fm, fnames, kinds, err = recover_frame(bnet, seqs)
    net = FrameNet(bnet, Fm)
    codec = F.Codec(vocab_tokens)
    tok, idx = codec.encode_batch(seqs)
    variables, valid = TB.map_variables(net, seqs)
    outs = net.outputs(seqs)
    T = valid.shape[1]
    outv = np.full((len(seqs), T), np.nan)
    for i, o in enumerate(outs):
        for j, x in enumerate(o[:T]):
            try:
                outv[i, j] = codec.code[x] if (not codec.numeric and x in codec.code) else float(x)
            except (TypeError, ValueError):
                outv[i, j] = np.nan

    def out_score(v):
        return TB._bijection(v["value"], outv) if v["kind"] == "categorical" else TB._affine_r2(v["value"], outv)[0]
    out_id = max(variables, key=lambda v: (round(out_score(v), 6), v["k"]))["id"]
    defined, library = TB.define_variables(variables, valid, tok, idx, codec, max_len, out_id, outv, net)
    bp, out_match = TB.assemble(net, variables, defined, library, outv, valid)
    bp.output = out_id
    cert = None
    if os.environ.get("BOX_CERTIFY") == "1":
        # the box's own step 5: its checker on fresh inputs it did not build the blueprint from (same input rule:
        # unique tokens if every given input has unique tokens)
        from box.verify import Checker
        uniq = all(len(set(s_)) == len(s_) for s_ in seqs)
        held = TB.probe_inputs(vocab_tokens, max_len, 150, 1, unique=uniq)
        rep = Checker(net, held, seed=0, n_pairs=120, control_reps=300).certify(bp, run_alone_line=1.0 - 1e-9)
        cert = {"verdict": rep["verdict"], "checks": [(c[0], bool(c[1]), str(c[2])) for c in rep["checks"]]}
    if verbose:
        print(f"== {task['id']}: frame error {err:.1e}; sublayers {kinds}; {len(variables)} variables; output {out_id} "
              f"(match {out_match:.3f})", flush=True)
        for v in variables:
            dd = defined[v["id"]]
            print(f"   {v['id']} at {v['site']} ({v['kind']}, {len(v['dims'])}): {dd['status']} {dd['definition']} [{dd['score']}]", flush=True)
    vmap = {v["id"]: v for v in variables}
    parts = []
    for p in bp.parts:
        v = vmap[p.id]
        Bf = np.asarray(p.basis, float)
        Bo = Fm @ Bf
        if p.encoding == "categorical":
            dd = defined[p.id]
            sym = bool(dd.get("symbolic")) or dd.get("definition") == "tokens"
            vals = []
            for i in range(Bf.shape[1]):
                sel = (v["value"] == i) & valid
                cand = np.unique(np.asarray(dd["value"])[sel]) if dd.get("value") is not None and sel.any() else []
                c = float(cand[0]) if len(cand) == 1 else float(i)
                vals.append(codec.decode.get(c, c) if sym else (int(c) if float(c).is_integer() else c))
            enc = {"kind": "categorical", "values": vals, "null_below": 0.5}
        else:
            enc = {"kind": "numerical", "scale": 1.0, "offset": 0.0}
        part = {"id": p.id, "type": p.type, "vtype": "SOp",
                "location": {"site": p.site, "positions": "all", "map": {"class": "subspace", "basis": Bo.tolist()}},
                "encoding": enc, "upstream": list(p.upstream)}
        if p.type == "rule":
            part["definition"] = p.definition
            part["claims"] = [{"claim": "type", "value": "rule", "p": 0.9, "evidence": "statistical"}]
            part["evidence"] = {"type": "statistical"}
        parts.append(part)
    lib = {}
    for name, e in (library or {}).items():
        lib[name] = dict(e)
        if e.get("kind") == "operation":
            lib[name]["source"] = bench_source(name, e["source"])
        prov = e.get("provenance")
        lib[name]["provenance"] = {"run": f"box/bench_s0.py on {task['id']}",
                                   "evidence": prov if isinstance(prov, str) else "operation search (box/tracr_box.py TRANSFORMS)"}
    n = len(parts)
    shares = {t: sum(p["type"] == t for p in parts) / n for t in ("rule", "lookup", "irrelevant", "unresolved")}
    overview = (f"Recovered blind by the box (box/bench_s0.py). The network's output is part {out_id}. "
                + " ".join(f"{p['id']} at {p['location']['site']} = {p.get('definition', '(unresolved)')}." for p in parts))[:4000]
    blue = {"interface": "bench-accept/v1", "task": task["id"], "domain": task["domain"], "parts": parts,
            "output": out_id, "library": lib, "overview": {"text": overview, "predictor": f"@{out_id}"},
            "systems": [{"id": "all", "role": "the whole computation", "parts": [p["id"] for p in parts],
                         "fed_by": [], "feeds": []}],
            "ledger": {"shares": shares, "unresolved_by_site": {}}, "usage": {"gpu_hours": 0.0, "model_tokens": 0}}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "blueprint.json")
    json.dump(blue, open(path, "w"), indent=1, default=float)
    return path, {"frame_error": err, "sublayers": {str(k): v for k, v in kinds.items()}, "n_variables": len(variables),
                  "defined": sum(defined[v["id"]]["status"] == "defined" for v in variables), "output": out_id,
                  "output_match": out_match, "certify": cert}


class Box:
    """The bench's Box protocol (INTERFACE.md §7), S0 only for now."""

    def __init__(self, config=None):
        self.config = config or {}

    def explain(self, task_dir, out_dir):
        return explain(task_dir, out_dir, verbose=False)[0]


if __name__ == "__main__":
    if sys.argv[1] == "--draw":
        draw, out = sys.argv[2], sys.argv[3]
        summary = {}
        for net_id in sorted(os.listdir(draw)):
            if not net_id.startswith("S0-tracr"):
                continue
            try:
                _, info = explain(os.path.join(draw, net_id), os.path.join(out, net_id))
                summary[net_id] = info
            except Exception as ex:  # a network the box cannot handle yet is reported, not hidden
                import traceback
                traceback.print_exc()
                summary[net_id] = {"error": f"{type(ex).__name__}: {ex}"}
        json.dump(summary, open(os.path.join(out, "box_summary.json"), "w"), indent=1, default=str)
    else:
        print(explain(sys.argv[1], sys.argv[2]))
