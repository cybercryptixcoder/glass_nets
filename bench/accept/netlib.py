"""The bench's network loader (INTERFACE.md §2): `load_network(task_dir)` returns an object that runs the network and
intervenes at the task's named sites. The scorer uses its own frozen copy of this file; the box may use it too.

Formats (task.json "network.format"):
- "tracr_np": a transformer compiled by Tracr (no layer norm; softmax attention; ReLU MLPs; residual connections),
  run in numpy from `network/params.npz` and `network/meta.json`. Sites resid_0 (after the embeddings), then
  resid_<k> after each sublayer; each has shape [positions, d], position 0 is the BOS position. Outputs are decoded
  through the published readout (`readout` [d, k]) as categorical values or one number.
- "rnn_relu": AlgZoo's one-layer ReLU RNN, `network/weights.npz` (W_x, W_h, W_o, seq_len). Sites h1 .. hT, each
  [1, hidden], and out [1, classes]. The decision is the argmax of out.
- "transformer": the lab's general transformer without layer norm (`bench/accept/vendor/lab_nets.py`), weights in
  `network/weights.npz`. Sites embed, attn<l>.out, resid<l>.mid, mlp<l>.pre, mlp<l>.post, mlp<l>.out, resid<l>.post,
  each [positions, width], and out [n_out_positions, classes].

Edits are functions on a site's activation [N, *site shape] returning the same shape (`interface.Edit`).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, "vendor") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "vendor"))
import lab_nets  # noqa: E402  (frozen copy of lab/nets.py at b629f1e3)


def load_network(task_dir, task=None):
    task = task or json.load(open(os.path.join(task_dir, "task.json")))
    fmt = task["network"]["format"]
    path = os.path.join(task_dir, task["network"].get("path", "network"))
    if fmt == "tracr_np":
        return TracrNP(path, task)
    if fmt == "rnn_relu":
        return RNNReLU(path, task)
    if fmt == "transformer":
        return LabTransformer(path, task)
    raise ValueError(f"unknown network format {fmt}")


# ====================================================================================================== Tracr
class TracrNP:
    """Numpy forward pass of a Tracr-compiled model. Params use Tracr's names with '/' kept: token_embed, pos_embed,
    layer_<l>/attn/{query,key,value,linear}/{w,b}, layer_<l>/mlp/{linear_1,linear_2}/{w,b}; plus readout [d, k]."""

    def __init__(self, path, task):
        self.task = task
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        P = np.load(os.path.join(path, "params.npz"))
        self.P = {k: np.asarray(P[k], np.float64) for k in P.files}
        self.n_layers, self.H = int(self.meta["n_layers"]), int(self.meta["n_heads"])
        self.tok_id = {str(k): int(v) for k, v in self.meta["input_vocab"].items()}
        self.bos = self.meta["bos"]
        self.readout = self.P["readout"]
        self.out_kind = self.meta["readout_kind"]
        self.out_values = self.meta.get("readout_values")
        self.sites = task["sites"]
        self.site_names = [f"resid_{k}" for k in range(2 * self.n_layers + 1)]

    def _encode(self, tokens):
        n = max(len(s) for s in tokens) + 1
        ids = np.zeros((len(tokens), n), dtype=int)
        mask = np.zeros((len(tokens), n), dtype=bool)
        for i, s in enumerate(tokens):
            seq = [self.bos] + [str(t) for t in s]
            ids[i, :len(seq)] = [self.tok_id[t] for t in seq]
            mask[i, :len(seq)] = True
        return ids, mask

    def run(self, ctx, edits=None, param_edits=None, return_sites=False):
        edits, P = edits or {}, dict(self.P)
        if param_edits:
            P.update({k: np.asarray(v, np.float64) for k, v in param_edits.items()})
        tokens = ctx["tokens"]
        ids, mask = self._encode(tokens)
        T = ids.shape[1]
        h = P["token_embed"][ids] + P["pos_embed"][np.arange(T)][None]
        sites = {}

        def at(k, h):
            name = f"resid_{k}"
            if name in edits:
                h = _pad_edit(edits[name], h, self.meta["max_seq_len"] + 1)
            if return_sites:
                sites[name] = h
            return h
        h = at(0, h)
        for l in range(self.n_layers):
            p = f"layer_{l}"
            q = h @ P[f"{p}/attn/query/w"] + P[f"{p}/attn/query/b"]
            k_ = h @ P[f"{p}/attn/key/w"] + P[f"{p}/attn/key/b"]
            v = h @ P[f"{p}/attn/value/w"] + P[f"{p}/attn/value/b"]
            B_, T_, _ = q.shape
            q, k_, v = (a.reshape(B_, T_, self.H, -1) for a in (q, k_, v))
            logits = np.einsum("bthk,bshk->bhts", q, k_)
            logits = np.where(mask[:, None, None, :], logits, -1e30)
            logits = logits - logits.max(-1, keepdims=True)
            w = np.exp(logits)
            w = w / w.sum(-1, keepdims=True)
            att = np.einsum("bhts,bshk->bthk", w, v).reshape(B_, T_, -1)
            h = at(2 * l + 1, h + att @ P[f"{p}/attn/linear/w"] + P[f"{p}/attn/linear/b"])
            z = np.maximum(h @ P[f"{p}/mlp/linear_1/w"] + P[f"{p}/mlp/linear_1/b"], 0)
            h = at(2 * l + 2, h + z @ P[f"{p}/mlp/linear_2/w"] + P[f"{p}/mlp/linear_2/b"])
        raw = h @ self.readout
        outs = []
        for i, s in enumerate(tokens):
            row = raw[i, 1:len(s) + 1]
            if self.out_kind == "numerical":
                outs.append([float(r[0]) for r in row])
            else:
                outs.append([self.out_values[int(np.argmax(r))] for r in row])
        res = {"outputs": outs, "raw": raw}
        if return_sites:
            res["sites"] = sites
        return res


def _pad_edit(fn, h, T_full):
    """Apply an edit written for the site's full shape [N, T_full, d] to a batch padded to T <= T_full."""
    N, T, d = h.shape
    if T == T_full:
        return fn(h)
    full = np.zeros((N, T_full, d))
    full[:, :T] = h
    return fn(full)[:, :T]


# ====================================================================================================== RNN
class RNNReLU:
    def __init__(self, path, task):
        self.task = task
        W = np.load(os.path.join(path, "weights.npz"))
        self.W = {k: np.asarray(W[k], np.float64) for k in W.files}
        self.T = int(self.W["seq_len"].ravel()[0])
        self.sites = task["sites"]

    def run(self, ctx, edits=None, param_edits=None, return_sites=False):
        W = dict(self.W)
        if param_edits:
            W.update({k: np.asarray(v, np.float64) for k, v in param_edits.items()})
        hook = {k: (lambda a, f=f: f(a[:, None, :])[:, 0, :]) for k, f in (edits or {}).items()}
        acts = {} if return_sites else None
        raw = lab_nets.rnn_relu(W, np.asarray(ctx["x"], np.float64), hook, acts)
        res = {"outputs": [int(i) for i in raw.argmax(1)], "raw": raw}
        if return_sites:
            res["sites"] = {k: v[:, None, :] for k, v in acts.items()}
        return res


# ====================================================================================================== transformer
class LabTransformer:
    def __init__(self, path, task):
        self.task = task
        W = np.load(os.path.join(path, "weights.npz"))
        self.W = {k: np.asarray(W[k], np.float64) for k in W.files}
        self.sites = task["sites"]
        self.shape = {s["name"]: s["shape"] for s in self.sites}

    def run(self, ctx, edits=None, param_edits=None, return_sites=False):
        W = dict(self.W)
        if param_edits:
            W.update({k: np.asarray(v, np.float64) for k, v in param_edits.items()})
        x = np.asarray(ctx["x"]).astype(int)
        N = len(x)
        hook = {k: (lambda a, f=f, k=k: f(a.reshape(N, *self.shape[k])).reshape(N, -1)) for k, f in (edits or {}).items()}
        acts = {} if return_sites else None
        raw = lab_nets.transformer(W, x, hook, acts)
        n_cls = W["W_U"].shape[1]
        raw3 = raw.reshape(N, -1, n_cls)
        dec = raw3.argmax(-1)
        res = {"outputs": [int(r[0]) if r.shape[0] == 1 else [int(v) for v in r] for r in dec], "raw": raw}
        if return_sites:
            res["sites"] = {k: v.reshape(N, *self.shape[k]) for k, v in acts.items() if k in self.shape}
        return res


def transformer_sites(W):
    """Site specs for the lab transformer, from lab_nets.site_table, reshaped to [positions, width]."""
    P = W["W_pos"].shape[0]
    out = []
    for order, (name, width, desc, stage) in enumerate(lab_nets.site_table("transformer", W)):
        if name.endswith(".pattern"):
            continue
        if name == "out":
            n_pos = len(W["out_positions"]) if "out_positions" in W else P
            out.append({"name": "out", "shape": [n_pos, W["W_U"].shape[1]], "order": order, "value_offset": 0,
                        "stream": None, "description": desc})
        else:
            stream = "resid" if (name == "embed" or ".mid" in name or ".post" in name and name.startswith("resid")) else None
            out.append({"name": name, "shape": [P, width // P], "order": order, "value_offset": 0, "stream": stream,
                        "description": desc})
    return out


def rnn_sites(W):
    T, hid, n_out = int(np.asarray(W["seq_len"]).ravel()[0]), W["W_h"].shape[0], W["W_o"].shape[1]
    return ([{"name": f"h{t}", "shape": [1, hid], "order": t - 1, "value_offset": 0, "stream": None,
              "description": f"hidden state after reading input {t} of {T}"} for t in range(1, T + 1)]
            + [{"name": "out", "shape": [1, n_out], "order": T, "value_offset": 0, "stream": None,
                "description": "the network's output scores"}])


# ====================================================================================================== reading parts
def read_coords(act, loc, task, bp_dir="."):
    """Coordinates of a location in a site's activation [N, T, d]: [N, T, k] for positions "all", [N, k] otherwise."""
    m = loc["map"]
    pos = loc.get("positions", "all")
    site = next(s for s in task["sites"] if s["name"] == loc["site"])
    if m["class"] == "subspace":
        B = basis_of(m, site)
        if pos == "all":
            return act @ B
        if pos == "flat":
            return act.reshape(len(act), -1) @ B
        return act[:, pos[0]] @ B
    if m["class"] == "dictionary_feature":
        D = np.load(os.path.join(bp_dir, m["dictionary"]))
        f = int(m["feature"])
        enc = lambda a: np.maximum((a - D["b_dec"]) @ D["W_enc"][:, f] + D["b_enc"][f], 0.0)[..., None]  # noqa: E731
        if pos == "all":
            return enc(act)
        if pos == "flat":
            return enc(act.reshape(len(act), -1))
        return enc(act[:, pos[0]])
    raise NotImplementedError("param_component locations are read through param edits, not activations")


def basis_of(m, site):
    if "basis" in m:
        return np.asarray(m["basis"], np.float64)
    labels = site.get("labels") or []
    B = np.zeros((len(labels), len(m["labels"])))
    for j, l in enumerate(m["labels"]):
        B[labels.index(l), j] = 1.0
    return B


def write_edit(loc, task, new_coords, bp_dir="."):
    """An edit that replaces a location's coordinates with new_coords ([N, T, k] for "all", [N, k] otherwise),
    leaving the orthogonal complement untouched (subspaces), or moving along the decoder row (dictionary features)."""
    m = loc["map"]
    pos = loc.get("positions", "all")
    site = next(s for s in task["sites"] if s["name"] == loc["site"])
    shape = site["shape"]
    if m["class"] == "subspace":
        B = basis_of(m, site)

        def fn(a):
            a = a.copy()
            if pos == "all":
                T = min(a.shape[1], new_coords.shape[1])
                a[:, :T] = a[:, :T] - (a[:, :T] @ B) @ B.T + new_coords[:, :T] @ B.T
            elif pos == "flat":
                f = a.reshape(len(a), -1)
                f = f - (f @ B) @ B.T + new_coords @ B.T
                a = f.reshape(a.shape)
            else:
                a[:, pos[0]] = a[:, pos[0]] - (a[:, pos[0]] @ B) @ B.T + new_coords @ B.T
            return a
        return fn
    if m["class"] == "dictionary_feature":
        D = np.load(os.path.join(bp_dir, m["dictionary"]))
        f = int(m["feature"])
        dec = D["W_dec"][f]

        def fn(a):
            a = a.copy()
            cur = read_coords(a, loc, task, bp_dir)
            delta = (new_coords - cur)[..., 0]
            if pos == "all":
                a = a + delta[..., None] * dec
            elif pos == "flat":
                a = (a.reshape(len(a), -1) + delta[:, None] * dec).reshape(a.shape)
            else:
                a[:, pos[0]] = a[:, pos[0]] + delta[:, None] * dec
            return a
        return fn
    raise NotImplementedError("param_component edits")
