"""Mechanism atlas v1 (CHANGES A1; intent and certificate in docs/glass/ATLAS.md §2).

The route for fragile networks, where no short approximate account can be faithful:
  1. decompose the weights EXACTLY into rank-1 mechanisms (SPD, lab/spd.py);
  2. give each mechanism a readable TRIGGER, a shallow rule over task features and the network's own neuron states,
     whose held-out balanced accuracy at predicting "the mechanism is needed" (importance > 0.5) is its confidence;
  3. certify the whole: the network rebuilt from its alive mechanisms, each switched on by its trigger and run on its
     OWN state, must reproduce the network's decisions (rule-gated fidelity), reported next to the task rule.

Targets are one-layer ReLU RNNs: h_t = relu(W_x x_t + W_h h_(t-1)), out = W_o h_N (AlgZoo, or the planted known answer).
Transformers come in a later version (ATLAS.md, A5).

    a = build_atlas(algzoo_rnn("zoo_2nd_argmax_8_5"), spd_model=PATH or None, depth=5)
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text

import spd

LAB = Path(__file__).resolve().parent
FEATURES = "v2"   # A16: trigger feature set. 'v3' (experiment, behind --features v3): neuron values + the 3rd largest
MATRICES = ("W_x", "W_h", "W_o")
TRUSTED = 0.9      # a trigger is used to switch its mechanism only at held-out confidence >= 0.9 (the labelling bar used
GATE_TAU = 0.05    # A15 gate v2: a trusted trigger switches its mechanism OFF only where its tree's P(needed) < GATE_TAU
                   # (a wrong 'off' loses decisions; a wrong 'on' costs nothing, since all-alive is exact). Pre-registered.
                   # throughout the units program, U19(c) (iii)); below it the mechanism stays always on and is reported so


@dataclass
class RNNTarget:
    name: str
    N: int                                   # sequence length
    H: int                                   # hidden size
    d_in: int                                # input size per step
    weights: dict                            # {W_x: [H, d_in], W_h: [H, H], W_o: [d_out, H]} as [d_out, d_in] tensors
    sample: Callable                         # n -> torch [n, N, d_in]
    X: np.ndarray                            # domain inputs [n, N, d_in]
    task_rule: float | None                  # agreement of the task's own answer with the network (its accuracy)
    task_features: Callable                  # (x [B, N, d_in], t) -> (names, [B, F]) for the step that reads x_t
    out_scale: float = 1.0
    meta: dict = field(default_factory=dict)

    def forward(self, x, mm):
        h = torch.zeros(x.shape[0], self.H, dtype=x.dtype, device=x.device)
        for t in range(self.N):
            h = torch.relu(mm("W_x", x[:, t]) + mm("W_h", h))
        return mm("W_o", h) / self.out_scale


# ------------------------------------------------------------------ targets

def _scalar_seq_features(x, t):
    """Task features of step t (1-based) for scalar-sequence tasks (2nd argmax, argmedian)."""
    s = x[:, :, 0]; B = len(s); prev = s[:, :t - 1]; cur = s[:, t - 1]; upto = s[:, :t]
    mx_p = prev.max(1) if t > 1 else np.full(B, np.nan); s2_p = np.sort(prev, 1)[:, -2] if t > 2 else np.full(B, np.nan)
    mn_p = prev.min(1) if t > 1 else np.full(B, np.nan)
    rank = (upto > cur[:, None]).sum(1) + 1
    amax = upto.argmax(1); a2 = np.argsort(upto, 1)[:, -2] if t > 1 else np.zeros(B, int); amin = upto.argmin(1)
    med = np.median(upto, 1)
    f = {"step t": np.full(B, t), "x_t": cur, "rank of x_t so far (1 = largest)": rank,
         "x_t minus the running median": cur - med, "distance of x_t's rank from the middle": np.abs(rank - (t + 1) / 2),
         "x_t minus the previous max": np.nan_to_num(cur - mx_p), "x_t minus the previous 2nd": np.nan_to_num(cur - s2_p),
         "x_t minus the previous min": np.nan_to_num(cur - mn_p),
         "steps since the max": t - 1 - amax, "steps since the 2nd": np.where(t > 1, t - 1 - a2, -1), "steps since the min": t - 1 - amin}
    if FEATURES.startswith("v3"):                                # A16
        srt = np.sort(upto, 1); nan = np.full(B, np.nan)
        s3_p = np.sort(prev, 1)[:, -3] if t > 3 else nan; a3 = np.argsort(upto, 1)[:, -3] if t > 2 else np.zeros(B, int)
        f.update({"x_t minus the previous 3rd": np.nan_to_num(cur - s3_p), "steps since the 3rd": np.where(t > 2, t - 1 - a3, -1),
                  "gap between the max and the 2nd so far": srt[:, -1] - srt[:, -2] if t > 1 else np.zeros(B),
                  "gap between the 2nd and the 3rd so far": srt[:, -2] - srt[:, -3] if t > 2 else np.zeros(B)})
    return list(f), np.stack(list(f.values()), 1).astype(float)


def _scalar_class_role(x):
    """Role of each output class (a sequence position) on each input: the rank of the value there (1 = largest)."""
    v = x[:, :, 0]
    return (v[:, None, :] > v[:, :, None]).sum(2) + 1


def algzoo_rnn(target_id, variant="original", n=16000):
    import judge
    meta, W, X = judge.load_target(target_id, variant, 0)
    N = int(W["seq_len"][0]); H = W["W_h"].shape[0]
    Ws = {"W_x": torch.tensor(W["W_x"].T.copy(), dtype=torch.float64), "W_h": torch.tensor(W["W_h"].T.copy(), dtype=torch.float64),
          "W_o": torch.tensor(W["W_o"].T.copy(), dtype=torch.float64)}
    Xn = np.asarray(X[:n], dtype=np.float64)[:, :, None]
    with torch.no_grad():
        h = torch.zeros(len(Xn), H, dtype=torch.float64); xt = torch.tensor(Xn[:8000])
        h = torch.zeros(len(xt), H, dtype=torch.float64)
        for t in range(N):
            h = torch.relu(xt[:, t] @ Ws["W_x"].T + h @ Ws["W_h"].T)
        scale = float((h @ Ws["W_o"].T).std())
    t = RNNTarget(name=f"{target_id}|{variant}", N=N, H=H, d_in=1, weights=Ws,
                  sample=lambda k: torch.randn(k, N, 1, dtype=torch.float64), X=Xn,
                  task_rule=float(meta["test_acc"]) if variant == "original" else None,
                  task_features=_scalar_seq_features, out_scale=scale, meta={"target": target_id, "variant": variant})
    t.class_role = _scalar_class_role; t.role_name = "rank of the value at that position (1 = largest)"
    return t


def planted_rnn(K=5, N=6, p_on=0.03, n=16000):
    """The U19(b') known answer: sum K sparse features over N steps; W_x = W_h = W_o = I."""
    I = torch.eye(K, dtype=torch.float64)

    def sample(k, g=None):
        on = (torch.rand(k, N, K, dtype=torch.float64, generator=g) < p_on).double()
        return on * torch.rand(k, N, K, dtype=torch.float64, generator=g)
    X = sample(n, torch.Generator().manual_seed(123)).numpy()

    def feats(x, t):
        names = [f"feature {j} arrives now (x_t[{j}] > 0)" for j in range(K)]
        return names, (x[:, t - 1] > 0).astype(float)
    return RNNTarget(name="planted_rnn", N=N, H=K, d_in=K, weights={"W_x": I, "W_h": I, "W_o": I}, sample=sample, X=X,
                     task_rule=None, task_features=feats, meta={"target": "planted_rnn"})


# ------------------------------------------------------------------ transformer targets (A5)

class TransformerTarget:
    """Attention-only transformer, one head per layer, no LayerNorm (the AlgZoo longest-cycle family; the planted
    known answer). Matrices (names without dots, for torch): W_E [d, V] (on one-hot tokens), L<l>_W_Q/K/V [k, d],
    L<l>_W_O [d, k], W_U [classes, d]. W_pos is a fixed additive term (not decomposed). Rows of every matrix call are
    (sample, position) pairs; for importances, features and ablations they are ordered position-major (row = p*B + b)
    so that a 'call' is one position. Inputs X: one-hot [n, P, V]."""
    kind = "transformer"

    def __init__(self, name, weights, W_pos, layers, out_pos, X, sample, task_rule, features, meta):
        self.name, self.weights, self.W_pos, self.layers, self.out_pos = name, weights, W_pos, layers, list(out_pos)
        self.X, self.sample, self.task_rule, self.features, self.meta = X, sample, task_rule, features, meta
        self.P, self.V = X.shape[1], X.shape[2]; self.d = W_pos.shape[1]; self.out_scale = 1.0
        self.matrices = tuple(["W_E"] + [f"L{l}_W_{m}" for l in range(layers) for m in "QKVO"] + ["W_U"])

    def forward(self, x, mm):
        B, P, V = x.shape
        Wp = torch.as_tensor(self.W_pos[:P], dtype=x.dtype, device=x.device)
        e = mm("W_E", x.reshape(B * P, V)).reshape(B, P, -1) + Wp
        for l in range(self.layers):
            a = e.reshape(B * P, -1)
            q = mm(f"L{l}_W_Q", a).reshape(B, P, -1); kk = mm(f"L{l}_W_K", a).reshape(B, P, -1); v = mm(f"L{l}_W_V", a).reshape(B, P, -1)
            att = torch.softmax(q @ kk.transpose(1, 2) / math.sqrt(q.shape[-1]), -1)
            e = e + mm(f"L{l}_W_O", (att @ v).reshape(B * P, -1)).reshape(B, P, -1)
        return mm("W_U", e[:, self.out_pos].reshape(B * len(self.out_pos), -1)).reshape(B, -1) / self.out_scale

    # rows: the forward's own order is sample-major (b*P + p); the atlas works position-major (p*B + b)
    def _pm(self, a, B, n):
        return a.reshape(B, n, -1).transpose(1, 0, 2).reshape(B * n, -1)

    def _bm(self, a, B, n):
        return a.reshape(n, B, -1).transpose(1, 0, 2).reshape(B * n, -1)

    def _rows(self, name):
        return len(self.out_pos) if name == "W_U" else self.P

    def _feats(self, x, name):
        names, F = self.features(x)                                  # [B, P, F]
        B = len(x)
        if name == "W_U":
            F = F[:, self.out_pos]
        return names, self._pm(F.reshape(B * self._rows(name), -1), B, self._rows(name))

    def network_calls(self, mods, x):
        B = len(x); rec = {}

        def mm(name, a):
            rec[name] = a
            return mods[name].apply(a)
        with torch.no_grad():
            self.forward(torch.tensor(x), mm)
            out = {}
            for k in self.matrices:
                n = self._rows(k); names, F = self._feats(x, k)
                G = mods[k].ci_pre(rec[k]).clamp(0, 1).numpy()
                out[k] = (names, F, self._pm(G, B, n))
        return out

    def run_gated(self, mods, x, gate):
        B = len(x); U = {k: mods[k].U.detach().numpy() for k in self.matrices}; Vv = {k: mods[k].V.detach().numpy() for k in self.matrices}

        def mm(name, a):
            a = a.numpy(); n = self._rows(name); names, F = self._feats(x, name)
            m = self._bm(gate(name, names, F, self._pm(a, B, n)), B, n)
            out = ((a @ Vv[name].T) * m) @ U[name]
            R = getattr(mods[name], "_residual", None)
            return torch.tensor(out + a @ R.T if R is not None else out)
        with torch.no_grad():
            return self.forward(torch.tensor(x), mm).numpy()

    def ablated(self, x, name=None, call=None, U=None, V=None):
        B = len(x)

        def mm(k, a):
            y = a @ self.weights[k].T
            if k == name:
                n = self._rows(k); rows = np.arange(B * n) % n == call          # sample-major rows at position `call`
                cut = (a.numpy() @ V)[:, None] * U[None]
                y = y - torch.tensor(cut * rows[:, None])
            return y
        with torch.no_grad():
            return self.forward(torch.tensor(x), mm).numpy()


def _lc_features(x):
    """Per (sample, position) features of a functional graph given as one-hot tokens: x_i = f(i)."""
    t = x.argmax(-1); B, P = t.shape; idx = np.arange(P)
    f = {"position i": np.broadcast_to(idx, (B, P)), "token f(i)": t, "f(i) = i (fixed point)": t == idx[None],
         "f(f(i))": np.take_along_axis(t, t, 1)}
    f["f(f(i)) = i (2-cycle)"] = f["f(f(i))"] == idx[None]
    f["number of j with f(j) = i"] = (t[:, :, None] == idx[None, None, :]).sum(1)
    f["number of j with f(j) = f(i)"] = (t[:, :, None] == t[:, None, :]).sum(2)
    cyc = np.zeros((B, P)); cur = t.copy()            # length of the cycle through i (0 if i is not on a cycle)
    for L in range(1, P + 1):
        hit = (cur == idx[None]) & (cyc == 0); cyc[hit] = L; cur = np.take_along_axis(t, cur, 1)
    f["length of the cycle through i (0 = not on one)"] = cyc
    f["longest cycle length (the task's answer + 1)"] = np.broadcast_to(cyc.max(1)[:, None], (B, P))
    names = list(f)
    return names, np.stack([np.asarray(v, float) for v in f.values()], -1)


def algzoo_transformer(target_id, variant="original", n=16000):
    import judge
    meta, W, X = judge.load_target(target_id, variant, 0)
    assert all(W[f"L{l}.W_Q"].shape[0] == 1 for l in range(2)), "one head per layer"
    L = sum(1 for k in W if k.endswith(".W_Q"))
    Ws = {"W_E": torch.tensor(W["W_E"].T.copy(), dtype=torch.float64), "W_U": torch.tensor(W["W_U"].T.copy(), dtype=torch.float64)}
    for l in range(L):
        for m in "QKV":
            Ws[f"L{l}_W_{m}"] = torch.tensor(W[f"L{l}.W_{m}"][0].T.copy(), dtype=torch.float64)      # [k, d]
        Ws[f"L{l}_W_O"] = torch.tensor(W[f"L{l}.W_O"][0].T.copy(), dtype=torch.float64)                # [d, k]
    toks = np.asarray(X, dtype=int)[:n]; V = W["W_E"].shape[0]; P = toks.shape[1]
    Xo = np.eye(V)[toks]

    def sample(k):
        return torch.nn.functional.one_hot(torch.randint(0, V, (k, P)), V).double()
    t = TransformerTarget(f"{target_id}|{variant}", Ws, np.asarray(W["W_pos"], dtype=np.float64), L,
                          np.asarray(W["out_positions"]).astype(int), Xo, sample,
                          float(meta["test_acc"]) if variant == "original" else None, _lc_features,
                          {"target": target_id, "variant": variant})
    with torch.no_grad():       # the reconstruction loss is relative: logits are divided by their spread (as for RNNs)
        t.out_scale = float(t.forward(torch.tensor(Xo[:4000]), lambda n_, a: a @ Ws[n_].T).std())
    return t


def planted_transformer(P=4, V=4, s=4.0, n=16000):
    """Known answer: one attention layer copies the PREVIOUS token to the last position, which the readout reads.
    Embedding: token one-hot (dims 0..V-1), position one-hot (V..V+P-1), an empty 'previous token' slot (V+P..V+P+V-1).
    Q/K select position i-1; V/O copy token v into the slot; W_U reads the slot. Planted mechanisms: one per token in
    W_E, W_V, W_O and W_U ('copy token v')."""
    d = V + P + V; k = max(P, V)
    WE = torch.zeros(d, V, dtype=torch.float64); WE[:V, :V] = torch.eye(V, dtype=torch.float64)
    Wpos = np.zeros((P, d)); Wpos[np.arange(P), V + np.arange(P)] = 1.0
    WQ = torch.zeros(k, d, dtype=torch.float64); WK = torch.zeros(k, d, dtype=torch.float64)
    for i in range(P):
        WQ[i, V + i] = s
        if i + 1 < P:
            WK[i + 1, V + i] = s
    WV = torch.zeros(k, d, dtype=torch.float64); WV[:V, :V] = torch.eye(V, dtype=torch.float64)
    WO = torch.zeros(d, k, dtype=torch.float64); WO[V + P:, :V] = torch.eye(V, dtype=torch.float64)
    WU = torch.zeros(V, d, dtype=torch.float64); WU[:, V + P:] = torch.eye(V, dtype=torch.float64)
    Ws = {"W_E": WE, "L0_W_Q": WQ, "L0_W_K": WK, "L0_W_V": WV, "L0_W_O": WO, "W_U": WU}
    g = torch.Generator().manual_seed(7)
    X = torch.nn.functional.one_hot(torch.randint(0, V, (n, P), generator=g), V).double().numpy()

    def sample(m):
        return torch.nn.functional.one_hot(torch.randint(0, V, (m, P)), V).double()

    def feats(x):
        t = x.argmax(-1); B = len(t); prev = np.concatenate([np.full((B, 1), -1), t[:, :-1]], 1)
        f = {"position i": np.broadcast_to(np.arange(P), (B, P))}
        for v in range(V):
            f[f"token at i is {v}"] = t == v
        for v in range(V):
            f[f"token at i-1 is {v}"] = prev == v
        return list(f), np.stack([np.asarray(a, float) for a in f.values()], -1)
    return TransformerTarget("planted_transformer", Ws, Wpos, 1, [P - 1], X, sample, None, feats, {"target": "planted_transformer"})


# ------------------------------------------------------------------ decomposition

def decompose(tgt, steps=30000, batch=512, seed=0, C=20, log=1000, device="cpu", dtype=torch.float32, coef_min=3e-3, recon="mse"):
    mods = spd.decompose(tgt.forward, tgt.weights, tgt.sample, C=C, steps=steps, coef_min=coef_min, p=1.0,
                         coef_faith=1.0, lr=1e-3, batch=batch, seed=seed, log=log, device=device, dtype=dtype,
                         recon=recon, kl_scale=getattr(tgt, "out_scale", 1.0))
    return mods.cpu().double()


def load_decomposition(tgt, path, C=None):
    d = torch.load(path, weights_only=False)
    if C is None:                                                 # the number of components is read from the saved U (A14: --C)
        C = next(v.shape[0] for k, v in d["mods"].items() if k.endswith(".U"))
    mods = torch.nn.ModuleDict({k: spd.Decomposed(v, C).double() for k, v in tgt.weights.items()})
    mods.load_state_dict({k: v.double() for k, v in d["mods"].items()})     # float32 decompositions load as float64 (A9)
    return mods


# ------------------------------------------------------------------ features, states, gated runs

def _features(tgt, x, t, h, end=False):
    """Features for the matrices that act at step t (W_x reads x_t, W_h reads h = h_(t-1)); end=True: W_o reads h = h_N."""
    names, F = tgt.task_features(x, t)
    lab = "in the final state" if end else "in the state before this step"
    if FEATURES == "v3":                                         # A16: neuron values (a split at 0 is 'active')
        return list(names) + [f"neuron {j} {lab}" for j in range(tgt.H)], np.concatenate([F, h], 1)
    names = list(names) + [f"neuron {j} active {lab}" for j in range(tgt.H)]
    F = np.concatenate([F, (h > 1e-9).astype(float)], 1)
    if FEATURES == "v3b":                                        # A16 variant: on/off AND values
        return names + [f"neuron {j} {lab}" for j in range(tgt.H)], np.concatenate([F, h], 1)
    return names, F


def set_residuals(mods, alive):
    """A12: each matrix keeps an always-on residual R = W - sum over alive mechanisms (the dead subcomponents' weight)."""
    with torch.no_grad():
        for k, m in mods.items():
            S = sum((torch.outer(m.U[c], m.V[c]) for c in alive[k]), torch.zeros_like(m.W))
            m._residual = (m.W - S).numpy()


def _apply(mods, name, a, mask):
    m = mods[name]
    out = ((a @ m.V.detach().numpy().T) * mask) @ m.U.detach().numpy()
    return out + a @ m._residual.T if hasattr(m, "_residual") else out


def run_gated(tgt, mods, x, gate):
    """The network rebuilt from mechanisms: gate(name, F_names, F, a) -> mask [B, C]; runs on its own state."""
    B = len(x); h = np.zeros((B, tgt.H))
    for t in range(1, tgt.N + 1):
        names, F = _features(tgt, x, t, h)
        h = np.maximum(_apply(mods, "W_x", x[:, t - 1], gate("W_x", names, F, x[:, t - 1]))
                       + _apply(mods, "W_h", h, gate("W_h", names, F, h)), 0)
    names, F = _features(tgt, x, tgt.N, h, end=True)
    return _apply(mods, "W_o", h, gate("W_o", names, F, h)) / tgt.out_scale


def _network_calls(tgt, mods, x):
    """Per matrix: the features at each call (network's own states) and the importances: {name: (names, F, G)}."""
    B = len(x); h = np.zeros((B, tgt.H)); Fx = []; Gx = []; Gh = []
    W = {k: v.numpy() for k, v in tgt.weights.items()}
    with torch.no_grad():
        ci = lambda k, a: mods[k].ci_pre(torch.tensor(a)).clamp(0, 1).numpy()   # noqa: E731
        for t in range(1, tgt.N + 1):
            names, F = _features(tgt, x, t, h); Fx.append(F); Gx.append(ci("W_x", x[:, t - 1])); Gh.append(ci("W_h", h))
            h = np.maximum(x[:, t - 1] @ W["W_x"].T + h @ W["W_h"].T, 0)
        names_o, Fo = _features(tgt, x, tgt.N, h, end=True); Go = ci("W_o", h)
    F = np.concatenate(Fx)
    return {"W_x": (names, F, np.concatenate(Gx)), "W_h": (names, F, np.concatenate(Gh)), "W_o": (names_o, Fo, Go)}


# ------------------------------------------------------------------ effects (A2)

def _net_ablated(tgt, x, name=None, call=None, U=None, V=None):
    """The network on x, with one rank-1 mechanism U V^T removed from matrix `name` at one call (None: intact)."""
    W = {k: v.numpy() for k, v in tgt.weights.items()}; h = np.zeros((len(x), tgt.H))
    cut = lambda a: (a @ V)[:, None] * U[None]          # noqa: E731
    for t in range(1, tgt.N + 1):
        ax = x[:, t - 1] @ W["W_x"].T; ah = h @ W["W_h"].T
        if name == "W_x" and call == t - 1:
            ax = ax - cut(x[:, t - 1])
        if name == "W_h" and call == t - 1:
            ah = ah - cut(h)
        h = np.maximum(ax + ah, 0)
    out = h @ W["W_o"].T
    if name == "W_o":
        out = out - cut(h)
    return out / tgt.out_scale


def effects(tgt, mods, x, calls, alive):
    """Per alive mechanism: remove it at one call at a time; compare the output change where it is needed
    (importance > 0.5) with where it is not (< 0.1). For transformers a 'call' is one position."""
    import judge
    abl = tgt.ablated if hasattr(tgt, "ablated") else (lambda x_, *a: _net_ablated(tgt, x_, *a))
    base = abl(x); B = len(x); res = {}
    roles = tgt.class_role(x) if getattr(tgt, "class_role", None) else None          # A7
    base_dec = judge.decide(base)[:, 0] if roles is not None else None
    for k in getattr(tgt, "matrices", MATRICES):
        G = calls[k][2]; n_calls = len(G) // B
        for c in alive[k]:
            U = mods[k].U[c].detach().numpy(); V = mods[k].V[c].detach().numpy()
            on_n, off_n, flips, d_on = [], [], [], np.zeros(base.shape[1]); trans = {}
            for call in range(n_calls):
                g = G[call * B:(call + 1) * B, c]
                if not (g > 0.5).any() and not (g < 0.1).any():
                    continue
                out = abl(x, k, call, U, V); d = out - base; nrm = np.linalg.norm(d, axis=1)
                on, off = g > 0.5, g < 0.1
                on_n.append(nrm[on]); off_n.append(nrm[off]); d_on += d[on].sum(0)
                fl = ~judge.same(base[on], out[on]) if on.any() else np.zeros(0, bool)
                flips.append(fl)
                if roles is not None and fl.any():
                    idx = np.where(on)[0][fl]; new_dec = judge.decide(out[idx])[:, 0]
                    for o, nw in zip(roles[idx, base_dec[idx]], roles[idx, new_dec]):
                        trans[(int(o), int(nw))] = trans.get((int(o), int(nw)), 0) + 1
            on_all = np.concatenate(on_n) if on_n else np.zeros(0); off_all = np.concatenate(off_n) if off_n else np.zeros(0)
            e_on = float(on_all.mean()) if len(on_all) else 0.0; e_off = float(off_all.mean()) if len(off_all) else 0.0
            mean_d = d_on / max(len(on_all), 1)
            res[(k, c)] = {"effect_when_needed": e_on, "effect_when_not_needed": e_off,
                           "specificity": float(min(e_on / (e_off + 1e-12), 1e6)),
                           "decision_flip_rate_when_needed": float(np.concatenate(flips).mean()) if flips and len(np.concatenate(flips)) else 0.0,
                           "mean_output_change_when_needed": mean_d.tolist(),
                           "most_lowered_class": int(mean_d.argmin()), "most_raised_class": int(mean_d.argmax())}
            if roles is not None:
                tot = sum(trans.values()) or 1
                res[(k, c)]["role_changes"] = [{"from": o, "to": nw, "share_of_flips": n_ / tot}
                                               for (o, nw), n_ in sorted(trans.items(), key=lambda kv: -kv[1])[:3]]
    return res


# ------------------------------------------------------------------ readable rules

def rule_text(tree, names):
    """The tree as a short readable rule: the conditions under which the mechanism is ON, with subtrees whose leaves
    all agree collapsed (sklearn keeps splits that don't change the class)."""
    t = tree.tree_
    cls = lambda i: bool(t.value[i][0].argmax() == list(tree.classes_).index(True)) if True in tree.classes_ else False  # noqa: E731

    def uniform(i):                     # None if mixed, else the class of every leaf below i
        if t.children_left[i] == -1:
            return cls(i)
        a, b = uniform(t.children_left[i]), uniform(t.children_right[i])
        return a if a is not None and a == b else None

    def paths(i, conds):               # conds: list of (feature, op, value); op in {"<=", ">", "not", "is"}
        u = uniform(i)
        if u is not None:
            if u:
                nodes.append(i)
            return [conds] if u else []
        f, thr = names[t.feature[i]], t.threshold[i]
        binary = set(np.unique(tree._fit_values[:, t.feature[i]])) <= {0.0, 1.0} if hasattr(tree, "_fit_values") else False
        lo, hi = ((f, "not", None), (f, "is", None)) if binary else ((f, "<=", thr), (f, ">", thr))
        return paths(t.children_left[i], conds + [lo]) + paths(t.children_right[i], conds + [hi])

    def text(conds):                    # merge conditions on the same feature: keep the tightest bounds
        out, seen = [], {}
        for f, op, v in conds:
            seen.setdefault(f, []).append((op, v))
        for f, cs in seen.items():
            if any(op in ("is", "not") for op, _ in cs):
                out.append(f"not ({f})" if cs[-1][0] == "not" else f); continue
            lo = max((v for op, v in cs if op == ">"), default=None); up = min((v for op, v in cs if op == "<="), default=None)
            if lo is not None and up is not None:
                out.append(f"{lo:.2f} < {f} <= {up:.2f}")
            elif lo is not None:
                out.append(f"{f} > {lo:.2f}")
            else:
                out.append(f"{f} <= {up:.2f}")
        return " AND ".join(out)
    nodes = []
    ps = paths(0, [])
    tree._features_used = sorted({f for p in ps for f, _, _ in p})        # A8
    if not ps:
        return "never"
    if ps == [[]]:
        return "always"
    tree._rule_full = "\n".join("ON when " + text(p) for p in ps)
    if not hasattr(tree, "_fit_values"):
        return tree._rule_full
    # readability: list paths by how many training inputs they switch on; show those covering >= 80%, fold the rest
    path_of = tree.decision_path(tree._fit_values)
    cover = np.array([path_of[:, nd].sum() for nd in nodes], dtype=float); order = np.argsort(-cover); tot = cover.sum() or 1.0
    shown, acc = [], 0.0
    for j in order:
        shown.append(j); acc += cover[j]
        if acc / tot >= 0.8:
            break
    rest = [j for j in order if j not in shown]
    lines = [f"ON when {text(ps[j])}" + (f" ({100 * cover[j] / tot:.0f}% of its ON cases)" if len(ps) > 1 else "") for j in shown]
    if rest:
        lines.append(f"+ {len(rest)} rarer case(s), {100 * cover[rest].sum() / tot:.0f}% of its ON cases")
    return "\n".join(lines)


# ------------------------------------------------------------------ the atlas

def _p_on(t, F):
    """P(needed) from a trigger: a constant trigger gives 0 or 1; a tree gives its (class-balanced) leaf probability."""
    if isinstance(t, bool):
        return np.full(len(F), float(t))
    cls = list(t.classes_)
    return t.predict_proba(F)[:, cls.index(True)] if True in cls else np.zeros(len(F))


def _trigger_on(t, F, tau=None):
    """v1 (tau None): the tree's own decision (P(needed) >= 0.5). v2: on unless P(needed) < tau."""
    if tau is None:
        return np.full(len(F), float(t)) if isinstance(t, bool) else t.predict(F).astype(float)
    return (_p_on(t, F) >= tau).astype(float)


def build_atlas(tgt, spd_model=None, depth=5, n=16000, steps=30000, batch=512, seed=0):
    import judge
    mods = load_decomposition(tgt, spd_model) if spd_model else decompose(tgt, steps, batch, seed)
    MATS = getattr(tgt, "matrices", MATRICES)
    calls = tgt.network_calls if hasattr(tgt, "network_calls") else (lambda m_, x_: _network_calls(tgt, m_, x_))
    gated = tgt.run_gated if hasattr(tgt, "run_gated") else (lambda m_, x_, g_: run_gated(tgt, m_, x_, g_))
    X = tgt.X[:n]; half = len(X) // 2; xtr, xte = X[:half], X[half:]
    tr, te = calls(mods, xtr), calls(mods, xte)
    C = mods[MATS[0]].U.shape[0]
    alive = {k: [c for c in range(C) if tr[k][2][:, c].mean() > 0.01] for k in MATS}
    atlas = {"target": tgt.meta, "kind": getattr(tgt, "kind", "rnn"), "task_rule": tgt.task_rule, "depth": depth, "matrices": {}}
    trees = {}
    for k in MATS:
        names, Ftr, Gtr = tr[k]; _, Fte, Gte = te[k]; mechs = []
        with torch.no_grad():
            faith = float((mods[k].W - mods[k].weight()).norm() / mods[k].W.norm())
            alive_err = float((mods[k].W - sum(torch.outer(mods[k].U[c], mods[k].V[c]) for c in alive[k])).norm() / mods[k].W.norm()) if alive[k] else 1.0
        for c in alive[k]:
            ytr, yte = Gtr[:, c] > 0.5, Gte[:, c] > 0.5
            clear = (Gte[:, c] > 0.9) | (Gte[:, c] < 0.1)           # A19: clear-cut calls (the rest are graded)

            def _bal(p_, y_):
                return 0.5 * (p_[y_].mean() + (~p_[~y_]).mean()) if y_.any() and (~y_).any() else float((p_ == y_).mean()) if len(y_) else 1.0
            if ytr.all() or (~ytr).all():
                const = bool(ytr.all()); trees[(k, c)] = const
                trig = {"rule": "always" if const else "never", "confidence": float((yte == const).mean()),
                        "confidence_clear": float((yte[clear] == const).mean()) if clear.any() else 1.0}
            else:
                tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Ftr, ytr)
                tree._fit_values = Ftr
                p = tree.predict(Fte)
                conf = 0.5 * (p[yte].mean() + (~p[~yte]).mean()) if yte.any() and (~yte).any() else float((p == yte).mean())
                trees[(k, c)] = tree
                trig = {"rule": rule_text(tree, names), "confidence": float(conf), "tree": export_text(tree, feature_names=names, decimals=2),
                        "confidence_clear": float(_bal(p[clear], yte[clear]))}
                trig["features_used"] = tree._features_used; trig["rule_full"] = getattr(tree, "_rule_full", trig["rule"])
                task_cols = [i for i, nm in enumerate(names) if not nm.startswith("neuron ")]          # A21
                if not any(f_.startswith("neuron ") for f_ in tree._features_used):
                    trig["task_level"] = {"rule": trig["rule"], "confidence": trig["confidence"], "confidence_clear": trig["confidence_clear"],
                                          "features_used": trig["features_used"]}
                elif task_cols:
                    tn = [names[i] for i in task_cols]
                    t2 = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Ftr[:, task_cols], ytr)
                    t2._fit_values = Ftr[:, task_cols]
                    p2 = t2.predict(Fte[:, task_cols])
                    c2 = 0.5 * (p2[yte].mean() + (~p2[~yte]).mean()) if yte.any() and (~yte).any() else float((p2 == yte).mean())
                    trig["task_level"] = {"rule": rule_text(t2, tn), "confidence": float(c2), "confidence_clear": float(_bal(p2[clear], yte[clear])),
                                          "features_used": t2._features_used}
            trig["graded_share"] = float(1 - clear.mean())
            m = mods[k]
            mechs.append({"id": f"{k}.{c}", "in_use": float(yte.mean()), "U": m.U[c].tolist(), "V": m.V[c].tolist(), "trigger": trig})
        atlas["matrices"][k] = {"mechanisms": mechs, "faithfulness_rel_error": faith, "alive_only_rel_error": alive_err}

    am = {k: np.isin(np.arange(C), alive[k]).astype(float) for k in MATS}
    set_residuals(mods, alive)                                   # A12
    with torch.no_grad():
        net = tgt.forward(torch.tensor(xte), lambda name, a: mods[name].apply(a)).numpy()
        rec = {}                                                 # A17: each matrix's real inputs on the held-out half

        def _rec(name, a):
            rec.setdefault(name, []).append(a.numpy()); return a @ mods[name].W.T
        tgt.forward(torch.tensor(xte), _rec)
    for k in MATS:
        A_ = np.concatenate(rec[k]); Wk = mods[k].W.numpy()
        atlas["matrices"][k]["alive_only_rel_error_on_data"] = float(np.linalg.norm(A_ @ mods[k]._residual.T) / (np.linalg.norm(A_ @ Wk.T) + 1e-12))

    conf = {(k, int(x["id"].rsplit(".", 1)[1])): x["trigger"]["confidence"] for k in MATS for x in atlas["matrices"][k]["mechanisms"]}
    conf_clear = {(k, int(x["id"].rsplit(".", 1)[1])): x["trigger"]["confidence_clear"] for k in MATS for x in atlas["matrices"][k]["mechanisms"]}

    def g_rules(only=None, trusted=None, tau=None, cf=None):
        """trusted = a confidence bar: a mechanism whose trigger is below it is kept always on (its trigger is not used).
        tau (A15 gate v2): switch off only where the trigger's P(needed) < tau. cf: which confidence the bar reads (A19)."""
        cf = conf if cf is None else cf

        def gate(k, names, F, a):
            m = np.zeros((len(F), C))
            if only is not None and k not in only:
                return m + am[k]
            for c in alive[k]:
                t = trees[(k, c)]
                if trusted is not None and cf[(k, c)] < trusted:
                    m[:, c] = 1.0
                else:
                    m[:, c] = _trigger_on(t, F, tau)
            return m
        return gate

    def g_ci(binary):
        def gate(k, names, F, a):
            with torch.no_grad():
                z = mods[k].ci_pre(torch.tensor(a)).clamp(0, 1).numpy()
            return ((z > 0.5).astype(float) if binary else z) * am[k]
        return gate
    fid = {}
    variants = [("all_alive", lambda k, n_, F, a: np.repeat(am[k][None], len(F), 0)), ("importance_masked", g_ci(False)),
                ("importance_binary", g_ci(True)), ("rule_gated", g_rules())]
    if "W_h" in MATS:
        variants.append(("rule_gated_W_h_only", g_rules({"W_h"})))
    variants.append(("rule_gated_trusted", g_rules(trusted=TRUSTED)))
    variants.append(("rule_gated_trusted_v2", g_rules(trusted=TRUSTED, tau=GATE_TAU)))      # A15
    variants.append(("rule_gated_trusted_v3", g_rules(trusted=TRUSTED, tau=GATE_TAU, cf=conf_clear)))   # A19
    for label, gate in variants:
        fid[label] = float(judge.same(net, gated(mods, xte, gate)).mean())
    # A17: the mechanisms alone (every alive mechanism on, the residual removed), in all matrices and in one at a time
    all_on = lambda k, n_, F, a: np.repeat(am[k][None], len(F), 0)          # noqa: E731
    saved = {k: mods[k]._residual for k in MATS}

    def _alone(only):
        for k in MATS:
            mods[k]._residual = np.zeros_like(saved[k]) if only in (None, k) else saved[k]
        v = float(judge.same(net, gated(mods, xte, all_on)).mean())
        for k in MATS:
            mods[k]._residual = saved[k]
        return v
    fid["mechanisms_alone"] = _alone(None)
    for k in MATS:
        atlas["matrices"][k]["mechanisms_alone_fidelity"] = _alone(k)
    atlas["fidelity"] = fid
    work = {}                                                    # A15: does the gate do work? (full network's held-out states)
    for v, tau, cf in (("v1", None, conf), ("v2", GATE_TAU, conf), ("v3", GATE_TAU, conf_clear)):
        off, offn, n_all, n_not = 0, 0, 0, 0
        for k in MATS:
            _, Fte, Gte = te[k]
            for c in alive[k]:
                if cf[(k, c)] < TRUSTED:
                    continue
                on = _trigger_on(trees[(k, c)], Fte, tau) > 0.5
                not_needed = Gte[:, c] < 0.1
                off += int((~on).sum()); n_all += len(on)
                offn += int((~on & not_needed).sum()); n_not += int(not_needed.sum())
        work[v] = {"off_share": off / n_all if n_all else 0.0, "off_recall": offn / n_not if n_not else 0.0}
    atlas["gate"] = {"tau": GATE_TAU, **work}
    atlas["_trees"], atlas["_alive"], atlas["_conf"] = trees, alive, conf      # kept for the playground (A10); not in JSON
    atlas["_conf_clear"] = conf_clear
    eff = effects(tgt, mods, xte, te, alive)                     # A2
    for k in MATS:
        for x in atlas["matrices"][k]["mechanisms"]:
            x["effect"] = eff[(k, int(x["id"].rsplit(".", 1)[1]))]
    return atlas


def summary(a):
    lines = [f"{a['target']}: task rule {a['task_rule']}, fidelity " + ", ".join(f"{k} {v:.3f}" for k, v in a["fidelity"].items())]
    for k, m in a["matrices"].items():
        confs = [x["trigger"]["confidence"] for x in m["mechanisms"]]
        lines.append(f"  {k}: {len(confs)} mechanisms, trigger confidence " + (f"min {min(confs):.3f} median {np.median(confs):.3f}, "
                     f"{sum(c >= 0.9 for c in confs)}/{len(confs)} >= 0.9" if confs else "-") + f"; alive-only rel. error {m['alive_only_rel_error']:.3f}")
        sp = [x["effect"]["specificity"] for x in m["mechanisms"] if "effect" in x]
        if sp:
            lines.append(f"      effects: specificity median {np.median(sp):.1f} (min {min(sp):.1f}); decision flips when needed: median "
                         f"{np.median([x['effect']['decision_flip_rate_when_needed'] for x in m['mechanisms']]):.3f}")
    return "\n".join(lines)


def _conf(x):
    """A22: the confidence the certificate trusts: clear-cut (A19), or the all-calls one for atlases built before A19."""
    return x["trigger"].get("confidence_clear", x["trigger"]["confidence"])


def _fid(a):
    """A22: the certificate's rebuilt-network fidelity: v3 (A15 gate, A19 trust), or older versions if absent."""
    f = a["fidelity"]
    return f.get("rule_gated_trusted_v3", f.get("rule_gated_trusted_v2", f["rule_gated_trusted"]))


def findings(a, control=None):
    """What the atlas says, as numbered statements (CHANGES A8). A trusted trigger has confidence >= TRUSTED."""
    def stats(at):
        mech = [x for m in at["matrices"].values() for x in m["mechanisms"]]
        trusted = [x for x in mech if _conf(x) >= TRUSTED and x["trigger"]["rule"] not in ("always", "never")]
        task_vars = {}
        for x in trusted:
            for f in x["trigger"].get("features_used", []):
                if not f.startswith("neuron "):
                    task_vars[f] = task_vars.get(f, 0) + 1
        uses_task = sum(1 for x in trusted if any(not f.startswith("neuron ") for f in x["trigger"].get("features_used", [])))
        groups = {}
        for x in mech:
            rc = x.get("effect", {}).get("role_changes") or []
            if rc and rc[0]["share_of_flips"] >= 0.4 and x["effect"]["decision_flip_rate_when_needed"] >= 0.05:
                groups.setdefault((rc[0]["from"], rc[0]["to"]), []).append(x["id"])
        na = [x for x in mech if x["trigger"]["rule"] not in ("always", "never")]
        n_task = sum(1 for x in na if x["trigger"].get("task_level", {}).get("confidence_clear", 0) >= TRUSTED)     # A21
        return {"n": len(mech), "n_trusted": len(trusted), "uses_task": uses_task, "task_vars": task_vars, "groups": groups,
                "n_switched": len(na), "n_task_expl": n_task,
                "always": sum(1 for x in mech if x["trigger"]["rule"] == "always"), "fid": _fid(at), "task_rule": at.get("task_rule")}
    S = stats(a); out = []
    out.append(f"1. The weights split exactly into {S['n']} mechanisms; {S['n_trusted']} have a trusted trigger (held-out confidence >= {TRUSTED} on clear-cut cases), "
               f"{S['always']} are always needed.")
    if S["n_switched"]:
        out.append(f"1b. {S['n_task_expl']} of the {S['n_switched']} mechanisms that are not always on have a task-level explanation "
                   f"(a trigger in task variables only, confidence >= {TRUSTED} on clear-cut cases).")
    if S["n_trusted"]:
        top = ", ".join(f"{k} ({v})" for k, v in sorted(S["task_vars"].items(), key=lambda kv: -kv[1])[:5]) or "none"
        out.append(f"2. {S['uses_task']} of the {S['n_trusted']} trusted triggers use a task variable (not only neuron states). Most used: {top}.")
    for (fr, to), ids in sorted(S["groups"].items(), key=lambda kv: -len(kv[1])):
        out.append(f"3. {len(ids)} mechanism(s) ({', '.join(ids)}) keep the network from answering rank {to} instead of rank {fr}: "
                   f"removing them where needed mostly flips the answer from rank {fr} to rank {to}.")
    out.append(f"4. Rebuilt from mechanisms switched by trusted triggers: {100 * S['fid']:.1f}% of decisions"
               + (f" (task rule {100 * S['task_rule']:.1f}%)." if S["task_rule"] is not None else "."))
    C = None
    if control is not None:
        C = stats(control)
        out.append(f"5. Shuffled-weights control: {C['n']} mechanisms, {C['n_trusted']} trusted triggers, {C['uses_task']} using a task variable "
                   f"(trained: {S['n']}, {S['n_trusted']}, {S['uses_task']}); functional groups: {sum(len(v) for v in C['groups'].values())} mechanisms "
                   f"(trained: {sum(len(v) for v in S['groups'].values())}).")
    return {"statements": out, "groups": {f"{fr}->{to}": ids for (fr, to), ids in S["groups"].items()}, "stats": {k: v for k, v in S.items() if k != "groups"},
            "control": None if C is None else {k: v for k, v in C.items() if k != "groups"}}


def _signature(x):
    e = x.get("effect") or {}; rc = e.get("role_changes") or []
    role = (rc[0]["from"], rc[0]["to"]) if rc and rc[0]["share_of_flips"] >= 0.4 and e.get("decision_flip_rate_when_needed", 0) >= 0.05 else None
    vars_ = frozenset(f for f in x["trigger"].get("features_used", []) if not f.startswith("neuron ")) \
        if _conf(x) >= TRUSTED else frozenset()
    return x["id"].rsplit(".", 1)[0], role, vars_


def compare_atlases(a, b, bar=0.5):
    """Recurring mechanisms between two atlases (CHANGES A11). A mechanism is 'functional' when its removal mostly
    flips the answer between two specific ranks. Similarity = same matrix role and same dominant role change, times the
    Jaccard overlap of the task variables its trusted trigger uses (1.0 when neither uses any)."""
    A = [(x["id"], _signature(x)) for m in a["matrices"].values() for x in m["mechanisms"]]
    B = [(x["id"], _signature(x)) for m in b["matrices"].values() for x in m["mechanisms"]]
    fa = [(i, s) for i, s in A if s[1] is not None]; fb = [(i, s) for i, s in B if s[1] is not None]

    def sim(s, t):
        if s[0] != t[0] or s[1] != t[1]:
            return 0.0
        u = s[2] | t[2]
        return 1.0 if not u else len(s[2] & t[2]) / len(u)
    cand = sorted(((sim(s, t), i, j) for i, s in fa for j, t in fb), reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for sc, i, j in cand:
        if sc >= bar and i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append({"a": i, "b": j, "similarity": sc})
    share_a = len(used_a) / len(fa) if fa else 0.0; share_b = len(used_b) / len(fb) if fb else 0.0
    return {"pairs": pairs, "functional_a": len(fa), "functional_b": len(fb), "share_a": share_a, "share_b": share_b,
            "summary": f"{len(pairs)} recurring mechanisms: {100 * share_a:.0f}% of A's {len(fa)} functional mechanisms "
                       f"and {100 * share_b:.0f}% of B's {len(fb)} have a counterpart with the same role and trigger variables."}


def fact_sheet(a):
    """The facts a narrative may use (A13)."""
    meta = a["target"]; L = [f"Network: {meta.get('target')} ({a.get('kind', 'rnn')})."]
    try:
        import judge  # noqa: F401
        m = json.loads((LAB / "models" / meta["target"] / "meta.json").read_text()); L.append(f"Task: {m['answer']['short']}.")
    except Exception:  # noqa: BLE001
        pass
    L += (a.get("findings") or findings(a))["statements"]
    for k, mm in a["matrices"].items():
        for x in sorted(mm["mechanisms"], key=lambda x: -x["in_use"]):
            if _conf(x) < TRUSTED:
                continue
            main = x["trigger"]["rule"].split("\n")[0]
            e = x.get("effect") or {}; rc = e.get("role_changes") or []
            eff = (f"; removing it where needed flips {100 * e['decision_flip_rate_when_needed']:.0f}% of decisions"
                   + (f", mostly from rank {rc[0]['from']} to rank {rc[0]['to']}" if rc else "")) if e else ""
            L.append(f"Mechanism {x['id']} (in use {100 * x['in_use']:.0f}% of the time, trigger confidence {_conf(x):.2f}): {main}{eff}.")
    return "\n".join(L)


def narrative_violations(text, facts):
    import re
    nums = set(re.findall(r"\d+(?:\.\d+)?", facts)); ids = set(re.findall(r"\b(?:W_[a-zA-Z]+|L\d_W_[A-Z]|W_[EU])\.\d+\b", facts))
    v = [f"number {n} not in the facts" for n in re.findall(r"\d+(?:\.\d+)?", text) if n not in nums]
    v += [f"mechanism {i} not in the facts" for i in re.findall(r"\b(?:W_[a-zA-Z]+|L\d_W_[A-Z]|W_[EU])\.\d+\b", text) if i not in ids]
    return v


def narrate(a):
    """One plain-English paragraph from the fact sheet, by the reader model, guarded for numbers and mechanism ids."""
    import os
    env = LAB / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
    import luminal
    facts = fact_sheet(a)
    prompt = ("Write ONE plain-English paragraph of at most 150 words that explains how this neural network solves its task, "
              "for a reader who knows the task but not the network. Use ONLY the facts below: do not add any number, mechanism "
              "id or claim that is not stated there. Refer to mechanisms by their ids. Where the facts leave something "
              "unexplained, say so.\n\nFACTS:\n" + facts)
    text, viol = "", ["not run"]
    for _ in range(2):
        r = luminal.complete({"model": luminal.model(), "messages": [{"role": "user", "content": prompt}], "max_tokens": 900})
        text = (r["choices"][0]["message"].get("content") or "").strip()
        viol = narrative_violations(text, facts)
        if not viol:
            break
        prompt += "\n\nYour previous paragraph broke the rule (" + "; ".join(viol[:5]) + "). Write it again using only the facts."
    return {"text": text, "violations": viol, "note": "written by the reader from the facts above; checked only for numbers and names"}


def gloss(a):
    """A20: add `rule_glossed` to every trigger: the rule with each neuron's unit-map label (results/glass/units/maps/<t>.json,
    per_unit) after its name, e.g. 'neuron 1 [≈ relu(max - x_t)] active'. Unexplained or unmapped neurons stay bare."""
    import re
    tid = (a.get("target") or {}).get("target"); var = (a.get("target") or {}).get("variant", "original")
    f = LAB.parent / "results" / "glass" / "units" / "maps" / f"{tid}.json"
    labels = {}
    if tid and var == "original" and f.exists():
        labels = {int(k): v.split(": ", 1)[-1] for k, v in json.loads(f.read_text())["map"]["per_unit"].items() if v != "unexplained"}

    def sub(m):
        j = int(m.group(1))
        return f"neuron {j} [≈ {labels[j]}]" if j in labels else m.group(0)
    for m in a["matrices"].values():
        for x in m["mechanisms"]:
            x["trigger"]["rule_glossed"] = re.sub(r"neuron (\d+)", sub, x["trigger"]["rule"])
    return a


def _expl(x):
    """A21: the explanation shown for a mechanism: its task-level trigger when trusted, else its (glossed) trigger."""
    tl = x["trigger"].get("task_level")
    if tl and tl.get("confidence_clear", 0) >= TRUSTED and x["trigger"]["rule"] not in ("always", "never"):
        return "task-level: " + tl["rule"] + (f" (clear-cut confidence {tl['confidence_clear']:.2f})" if tl["rule"] != x["trigger"]["rule"] else "")
    return x["trigger"].get("rule_glossed", x["trigger"]["rule"])


def render(a, control=None):
    """The atlas certificate in markdown (CHANGES A4)."""
    f = a["fidelity"]; tr = a["task_rule"]
    L = [f"# Mechanism atlas: {a['target'].get('target')} ({a['target'].get('variant', '')})", "",
         "The network's weights are split EXACTLY into rank-1 mechanisms (SPD). Each mechanism has a readable trigger "
         "(when it is needed), scored on held-out inputs (its confidence), and an effect (what removing it changes).", "",
         "## Whole-network certificate", "",
         f"- Rebuilt from its mechanisms, each switched on by its trigger where the trigger is trusted (confidence >= {TRUSTED} on clear-cut cases; "
         f"otherwise always on), and switched off only where the trigger gives it under a {100 * GATE_TAU:.0f}% chance of being needed: "
         f"keeps **{100 * _fid(a):.1f}%** of the network's decisions"
         + (f"; the task rule alone keeps {100 * tr:.1f}%." if tr is not None else "."),
         (f"- The mechanisms alone (all on, the small leftover residual removed): {100 * f['mechanisms_alone']:.1f}%. " if "mechanisms_alone" in f else "- ")
         + f"Every trigger used (no confidence bar): {100 * f['rule_gated']:.1f}%. Mechanisms switched by their own learned importances: {100 * f['importance_masked']:.1f}%. "
         f"Earlier gates, for the record: v1 {100 * f['rule_gated_trusted']:.1f}%"
         + (f", v2 {100 * f['rule_gated_trusted_v2']:.1f}%." if "rule_gated_trusted_v2" in f else "."), "",
         "## Findings", ""] + findings(a, control)["statements"] + [""] + (
         ["## In plain English", "", a["narrative"]["text"], "", f"*({a['narrative']['note']}"
          + (f"; unresolved: {'; '.join(a['narrative']['violations'])}" if a["narrative"]["violations"] else "") + ")*", ""]
         if a.get("narrative") else [])
    for k, m in a["matrices"].items():
        mech = m["mechanisms"]; confs = [_conf(x) for x in mech]
        L += [f"## {k}: {len(mech)} mechanisms ({sum(c >= TRUSTED for c in confs)} with a trusted trigger)", "",
              f"Alive mechanisms reproduce {k} to a relative error of {m['alive_only_rel_error']:.3f}.", "",
              "| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |", "|---|---|---|---|---|---|"]
        for x in sorted(mech, key=lambda x: -x["in_use"]):
            e = x.get("effect", {})
            eff = (f"changes the output by {e['effect_when_needed']:.2f} and flips {100 * e['decision_flip_rate_when_needed']:.0f}% of decisions; "
                   f"lowers class {e['most_lowered_class']}, raises class {e['most_raised_class']}") if e else "-"
            if e and e.get("role_changes"):
                eff += "; flips go " + ", ".join(f"rank {r['from']} → rank {r['to']} ({100 * r['share_of_flips']:.0f}%)" for r in e["role_changes"][:2])
            spec = "-" if not e else ("only where needed" if e["specificity"] >= 1e5 else f"{e['specificity']:.0f}x")
            L.append(f"| {x['id']} | {100 * x['in_use']:.0f}% | {_expl(x).replace(chr(10), '<br>')} | {_conf(x):.2f} | {eff} | {spec} |")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ playground (A10)

_PLAY = {}


def judge_decide(out):
    import judge
    return judge.decide(out)[:, 0]


def trusted_gate(trees, alive, conf, C, tau=None):
    """The certificate's gate (A22: pass conf = clear-cut confidences and tau = GATE_TAU for v3)."""
    am = {k: np.isin(np.arange(C), v).astype(float) for k, v in alive.items()}

    def gate(k, names, F, a):
        m = np.zeros((len(F), C))
        for c in alive[k]:
            m[:, c] = 1.0 if conf[(k, c)] < TRUSTED else _trigger_on(trees[(k, c)], F, tau)
        return m * am[k]
    return gate


def _load_play(tid):
    if tid not in _PLAY:
        import pickle
        P = pickle.loads((LAB.parent / "results" / "glass" / "atlas" / f"{tid}_d4_model.pkl").read_bytes())
        tgt = make_target(tid); mods = load_decomposition(tgt, LAB.parent / P["spd"]); set_residuals(mods, P["alive"])
        C = mods[next(iter(mods))].U.shape[0]
        _PLAY[tid] = {"tgt": tgt, "mods": mods, "gate": trusted_gate(P["trees"], P["alive"], P.get("conf_clear", P["conf"]), C, tau=GATE_TAU if "conf_clear" in P else None), **P}
    return _PLAY[tid]


def play(tid, x=None, seed=None):
    """One input through the network and through the network rebuilt from trusted-trigger-switched mechanisms.
    Returns, per matrix call, the mechanisms switched ON by their triggers and those the importances say are needed."""
    P = _load_play(tid); tgt, mods, gate = P["tgt"], P["mods"], P["gate"]
    if x is None:
        xi = tgt.X[np.random.default_rng(seed).integers(len(tgt.X))][None]
    elif getattr(tgt, "kind", "rnn") == "transformer":
        xi = np.eye(tgt.V)[np.asarray(x, dtype=int)][None]
    else:
        xi = np.asarray(x, dtype=np.float64).reshape(1, tgt.N, tgt.d_in)
    calls, count = [], {}

    def rec(k, names, F, a):
        m = gate(k, names, F, a)
        with torch.no_grad():
            g = mods[k].ci_pre(torch.tensor(a)).clamp(0, 1).numpy()
        for r in range(len(F)):
            i = count.get(k, 0); count[k] = i + 1
            label = (f"position {i}" if getattr(tgt, "kind", "rnn") == "transformer" else
                     ("end" if k == "W_o" else f"step {i + 1}"))
            calls.append({"matrix": k, "at": label, "on_by_trigger": [f"{k}.{c}" for c in np.where(m[r] > 0.5)[0]],
                          "on_by_importance": [f"{k}.{c}" for c in np.where(g[r] > 0.5)[0] if c in P["alive"][k]]})
        return m
    with torch.no_grad():
        net = tgt.forward(torch.tensor(xi), lambda name, a: mods[name].apply(a)).numpy()
    rebuilt = tgt.run_gated(mods, xi, rec) if hasattr(tgt, "run_gated") else run_gated(tgt, mods, xi, rec)
    shown = xi.argmax(-1)[0].tolist() if getattr(tgt, "kind", "rnn") == "transformer" else np.round(xi[0, :, 0] if tgt.d_in == 1 else xi[0], 4).tolist()
    return {"x": shown, "network_decision": int(judge_decide(net)[0]), "rebuilt_decision": int(judge_decide(rebuilt)[0]), "calls": calls}


def make_target(tid, variant="original"):
    if tid == "planted_rnn":
        return planted_rnn()
    if tid == "planted_transformer":
        return planted_transformer()
    if tid.startswith("lc_"):
        return algzoo_transformer(tid, variant)
    return algzoo_rnn(tid, variant)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a mechanism atlas (docs/glass/ATLAS.md).")
    ap.add_argument("target"); ap.add_argument("spd", nargs="?", help="saved SPD decomposition (.pt); omitted = decompose now")
    ap.add_argument("depth", nargs="?", type=int, default=4)
    ap.add_argument("--control-spd", help="SPD decomposition of the shuffled-weights network, for the control comparison")
    ap.add_argument("--narrate", action="store_true", help="add a guarded plain-English paragraph written by the reader model (A13)")
    ap.add_argument("--features", default="v2", choices=["v2", "v3", "v3b", "v3c"], help="trigger feature set (A16 experiments; outputs get _f3/_f3b/_f3c)")
    a_ = ap.parse_args()
    FEATURES = a_.features
    SUF = "_f" + FEATURES[1:] if FEATURES != "v2" else ""
    tgt = make_target(a_.target)
    a = build_atlas(tgt, spd_model=a_.spd, depth=a_.depth)
    ctrl = build_atlas(make_target(a_.target, "shuffled"), spd_model=a_.control_spd, depth=a_.depth) if a_.control_spd else None
    a["findings"] = findings(a, ctrl)
    gloss(a)                                                     # A20
    if a_.narrate:
        a["narrative"] = narrate(a)
    out = LAB.parent / "results" / "glass" / "atlas"; out.mkdir(parents=True, exist_ok=True)
    import pickle
    spd_rel = str(Path(a_.spd).resolve().relative_to(LAB.parent)) if a_.spd else None      # repo-relative path
    (out / f"{a_.target}_d{a_.depth}{SUF}_model.pkl").write_bytes(pickle.dumps({"spd": spd_rel,
        "trees": a.pop("_trees"), "alive": a.pop("_alive"), "conf": a.pop("_conf"), "conf_clear": a.pop("_conf_clear"), "depth": a_.depth}))
    if ctrl is not None:
        for k in ("_trees", "_alive", "_conf", "_conf_clear"):
            ctrl.pop(k, None)
    a["features"] = FEATURES
    (out / f"{a_.target}_d{a_.depth}{SUF}.json").write_text(json.dumps(a, indent=1))
    (out / f"{a_.target}_d{a_.depth}{SUF}.md").write_text(render(a, ctrl))
    if ctrl is not None:
        (out / f"{a_.target}_shuffled_d{a_.depth}{SUF}.json").write_text(json.dumps(ctrl, indent=1))
    print(summary(a)); print("\n".join(a["findings"]["statements"]))
