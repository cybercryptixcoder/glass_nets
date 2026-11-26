"""Rosetta engine: one round of the residual-driven loop on any network that exposes per-site unit activity.

define(units)      -> for each unit, the best short program (searched on training rows, scored on held-out rows)
line(controls)     -> the acceptance line: the largest held-out fidelity any control unit reaches
substitute(...)    -> put the program in place of the unit inside the network; share of the unit's effect restored
"""
import numpy as np

from rosetta.core import literal_specs, literal_matrix, search, search_pos, program_map, fidelity, describe


class Defs:
    def __init__(self, P, names, tr, te, n_search=1500, n_score=2500, seed=0):
        rng = np.random.default_rng(seed)
        self.P, self.names = P, names
        self.rs = np.sort(rng.choice(tr, min(n_search, len(tr)), replace=False))
        self.rt = np.sort(rng.choice(te, min(n_score, len(te)), replace=False))
        Ps = np.asarray(P[self.rs], dtype=np.float32)
        self.specs = literal_specs(Ps, names)
        self.L = literal_matrix(Ps, self.specs)
        self.L3 = self.L.reshape(len(self.specs), len(self.rs), -1)
        self.Pt = np.asarray(P[self.rt], dtype=np.float32)

    def define(self, unit_rows_fn, keep=5, center=True):
        """unit_rows_fn(rows) -> [len(rows), S] activity of one unit. Returns the training-best program and its
        held-out fidelity, plus runners-up (reported, never used to choose). center: remove the unit's fixed
        per-site mean (its spatial bias, the same for every input), so only input-dependent activity is defined."""
        ys = np.asarray(unit_rows_fn(self.rs), dtype=np.float32)
        yt = np.asarray(unit_rows_fn(self.rt), dtype=np.float32)
        if center:
            mu = ys.mean(0, keepdims=True)
            ys, yt = ys - mu, yt - mu
        ys, yt = ys.reshape(-1), yt.reshape(-1)
        S = ys.size // len(self.rs)
        if ys.std() < 1e-6:
            return {"program": None, "text": "(dead unit)", "train_corr": 0.0, "heldout_fidelity": 0.0, "runners_up": [],
                    "pos": {"program": None, "text": "(dead unit)", "train_corr": 0.0, "heldout_fidelity": 0.0}}
        res = search(ys, self.L, self.specs, keep=keep)
        out = []
        for prog, c in res:
            m = program_map(self.Pt, self.specs, prog)
            out.append({"program": [self.specs[i][:3] for i in prog], "text": describe(self.specs, prog),
                        "train_corr": round(c, 4), "heldout_fidelity": round(fidelity(yt, m), 4)})
        best = dict(out[0])
        best["runners_up"] = out[1:]
        # the same unit at the level the rest of the network can pool: its total per position
        ts, tt = ys.reshape(len(self.rs), S).sum(1), yt.reshape(len(self.rt), S).sum(1)
        rp = search_pos(ts, self.L3, self.specs, keep=3)
        prog, c = rp[0]
        m = program_map(self.Pt, self.specs, prog).reshape(len(self.rt), S).sum(1)
        best["pos"] = {"program": [self.specs[i][:3] for i in prog], "text": describe(self.specs, prog),
                       "train_corr": round(c, 4), "heldout_fidelity": round(fidelity(tt, m), 4)}
        return best


def spec_map(P, prog):
    """Program map from stored literal specs [(k, op, t), ...] on any primitive table P [n, S, K]."""
    X = np.asarray(P, dtype=np.float32).reshape(-1, P.shape[-1])
    b = np.ones(X.shape[0], dtype=np.float32)
    for k, op, t in prog:
        v = X[:, k]
        b *= (v >= t) if op == ">=" else (v < t) if op == "<" else (v <= t)
    return b.reshape(P.shape[0], P.shape[1])


def acceptance_line(control_defs, level="sq"):
    vals = [abs(d["heldout_fidelity"] if level == "sq" else d["pos"]["heldout_fidelity"]) for d in control_defs]
    return float(np.max(vals)) if vals else 1.0, float(np.quantile(vals, 0.99)) if vals else 1.0


def restore_share(out_orig, out_abl, out_sub):
    """1 - |sub - orig| / |abl - orig|, averaged where ablation matters."""
    da = np.abs(out_abl - out_orig)
    ds = np.abs(out_sub - out_orig)
    m = da > np.quantile(da, 0.5)
    return float(1 - ds[m].mean() / max(da[m].mean(), 1e-9))
