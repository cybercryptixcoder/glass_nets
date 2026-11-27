"""The checker v0 for networks whose sites are plain vectors and whose parts are numbers per input (AlgZoo RNNs and
other numeric-sequence networks; vocabulary domain `seq`). Same checks as box/verify.py: form, run alone,
interchange interventions against a family-wise random-direction line, substitution (reported), the lookup typing
rule. A part at site "input" is a function of the input alone, and a part at site "readout" applies the task's
decision rule to the network's outputs (argmax); neither is a variable inside the network, so neither is located."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "accept"))
import vocab  # noqa: E402

from box.blueprint import Blueprint, TYPES  # noqa: E402


class VecChecker:
    def __init__(self, net, X, test_idx=None, seed=0, n_pairs=400, control_reps=200):
        self.net, self.X = net, np.asarray(X, np.float64)
        self.rng = np.random.default_rng(seed)
        self.base = net.outputs(self.X)
        _, self.sites = net.run(self.X, return_sites=True)
        idx = np.arange(len(self.X)) if test_idx is None else np.asarray(test_idx)
        a = self.rng.choice(idx, n_pairs); b = self.rng.choice(idx, n_pairs)
        keep = a != b
        self.pa, self.pb = a[keep], b[keep]
        self.control_reps = control_reps
        self.site_names = net.sites_for(self.X.shape[1])

    def form(self, bp):
        if not isinstance(bp, Blueprint):
            return False, f"not a blueprint ({type(bp).__name__}): an account must be parts with vocabulary definitions"
        try:
            order = bp.order()
        except ValueError as e:
            return False, str(e)
        dom = vocab.get(bp.domain)
        types = {}
        for p in order:
            if p.type not in TYPES:
                return False, f"part {p.id}: type {p.type!r} not allowed"
            if p.site not in ("input", "readout"):
                if p.site not in self.site_names:
                    return False, f"part {p.id}: unknown site {p.site}"
                B = None if p.basis is None else np.asarray(p.basis, float).reshape(-1, 1)
                d = self.sites[p.site].shape[1]
                if B is None or B.shape[0] != d or not np.isclose(np.linalg.norm(B), 1.0):
                    return False, f"part {p.id}: location must be a unit direction in {p.site} (d={d}) with a scale"
            if p.type == "rule":
                ok, why, t = vocab.admissible(p.definition, dom, {u: types[u] for u in p.upstream}, bp.library) \
                    if isinstance(p.definition, str) else (False, "no definition", None)
                if not ok:
                    return False, f"part {p.id}: {why}"
                if t != p.vtype:
                    return False, f"part {p.id}: definition has type {t}, declared {p.vtype}"
            types[p.id] = p.vtype
        return (True, "ok") if bp.output in types else (False, "no output part")

    def _ops(self, bp):
        key = id(bp)
        if getattr(self, "_ops_key", None) != key:
            self._ops_key = key
            self._ops_code = {n: vocab.compile_operation(e["source"]) for n, e in (bp.library or {}).items()
                              if e.get("kind") == "operation"}
        return self._ops_code

    def values(self, bp, X, overrides=None):
        overrides = overrides or {}
        dom = vocab.get(bp.domain)
        ops = self._ops(bp)
        vals, types = {}, {}
        for p in bp.order():
            vals[p.id] = overrides[p.id] if p.id in overrides else \
                np.asarray(vocab.evaluate(p.definition, dom, {"x": X}, vals, types, bp.library, ops), np.float64)
            types[p.id] = p.vtype
        return vals

    def decision(self, bp, X, overrides=None):
        return np.rint(self.values(bp, X, overrides)[bp.output]).astype(int)

    def run_alone(self, bp, idx=None):
        idx = np.arange(len(self.X)) if idx is None else idx
        return float(np.mean(self.decision(bp, self.X[idx]) == self.base[idx]))

    def _swap(self, u, site, src_rows, sites=None):
        u = u.reshape(-1)
        sites = self.sites if sites is None else sites

        def fn(h):
            return h - np.outer(h @ u, u) + np.outer(sites[site][src_rows] @ u, u)
        return fn

    def _iia(self, bp, p, u, net=None, sites=None):
        """Agreement of the network's decision under a swap along u with the blueprint's predicted decision, on the
        pairs where the blueprint predicts that swapping part p changes the decision. net/sites: a control network."""
        net = self.net if net is None else net
        Xa = self.X[self.pa]
        vb = self.values(bp, self.X[self.pb])
        pred = self.decision(bp, Xa, overrides={p.id: vb[p.id]})
        own = self.decision(bp, Xa)
        dec = pred != own
        if not dec.any():
            return None, 0
        rows = np.nonzero(dec)[0]
        got = net.outputs(Xa[rows], {p.site: self._swap(u, p.site, self.pb[rows], sites)})
        before = net.outputs(Xa[rows])
        # a hit is a change, to the predicted value: a network that ignores the swap scores nothing
        return float(np.mean((got == pred[rows]) & (got != before))), len(rows)

    def control_lines(self, bp, located):
        """Family-wise lines, one per control family: the 99th percentile over repetitions of the best of K control
        scores, K = the number of parts tested. Families: random directions at the same site; the same direction in
        weight-shuffled copies of the network (each weight matrix's entries permuted)."""
        K = max(1, len(located))
        reps = self.control_reps // 10
        out = {}
        best = []
        for _ in range(reps):
            sc = []
            for _k in range(K):
                p = located[self.rng.integers(len(located))]
                v = self.rng.normal(size=self.sites[p.site].shape[1]); v /= np.linalg.norm(v)
                s_, _n = self._iia(bp, p, v)
                if s_ is not None:
                    sc.append(s_)
            if sc:
                best.append(max(sc))
        out["random directions"] = float(np.quantile(best, 0.99)) if best else 1.0
        best = []
        for _ in range(reps):
            sc = []
            for _k in range(K):
                p = located[self.rng.integers(len(located))]
                W = {k: self.rng.permutation(getattr(self.net, a).ravel()).reshape(getattr(self.net, a).shape)
                     for k, a in (("W_x", "Wx"), ("W_h", "Wh"), ("W_o", "Wo"))}
                cnet = type(self.net)(self.net.target, weights=W)
                _, csites = cnet.run(self.X, return_sites=True)
                s_, _n = self._iia(bp, p, np.asarray(p.basis, float), cnet, csites)
                if s_ is not None:
                    sc.append(s_)
            if sc:
                best.append(max(sc))
        out["shuffled-weight networks"] = float(np.quantile(best, 0.99)) if best else 1.0
        return out

    def certify(self, bp, run_alone_line):
        rep = {"name": bp.name if isinstance(bp, Blueprint) else "", "checks": []}
        ok, why = self.form(bp)
        rep["checks"].append(("form", ok, why))
        if not ok:
            rep["verdict"] = "rejected"; return rep
        ra = self.run_alone(bp)
        rep["checks"].append(("run alone", ra >= run_alone_line, f"agreement {ra:.4f} against line {run_alone_line:.4f}"))
        located = [p for p in bp.parts if p.site not in ("input", "readout") and p.type in ("rule", "lookup")]
        scores = {}
        for p in located:
            scores[p.id] = self._iia(bp, p, np.asarray(p.basis, float))
        lines = self.control_lines(bp, located)
        line = max(lines.values())
        rep["iia_line"] = line
        rep["control_lines"] = lines
        for p in located:
            s, n = scores[p.id]
            if s is None:
                rep["checks"].append((f"interventions {p.id}", p.type != "rule", "no decisive pairs: location untested"))
            elif p.type == "rule":
                rep["checks"].append((f"interventions {p.id}", s > line, f"IIA {s:.3f} on {n} decisive pairs, line {line:.3f}"))
            else:
                rep["checks"].append((f"typing {p.id}", s <= line, f"typed lookup; IIA {s:.3f} against line {line:.3f}"))
        rep["verdict"] = "certified" if all(c[1] for c in rep["checks"]) else "rejected"
        return rep


def fit_location(sites, site, values, idx, prefer_units=True):
    """A direction and scale mapping a site's activations to a part's values (an allowed linear map), fitted on the
    rows idx only. When one unit alone reproduces the values (R^2 > 0.9999), that unit is the location: on inputs
    where units move together (collinear on the data), least squares would otherwise spread the location over them,
    and interventions on that mixture test a different variable from the one the account names."""
    H = sites[site][idx]
    v = np.asarray(values)[idx]
    if prefer_units and np.var(v) > 0:
        best = None
        for k in range(H.shape[1]):
            h = H[:, k]
            if np.var(h) == 0:
                continue
            c = float(h @ v / (h @ h))
            r2 = 1 - np.var(v - c * h) / np.var(v)
            if r2 > 0.9999 and (best is None or r2 > best[0]):
                best = (r2, k, c)
        if best is not None:
            u = np.zeros(H.shape[1]); u[best[1]] = 1.0
            return u, best[2]
    w, *_ = np.linalg.lstsq(H, v, rcond=None)
    n = float(np.linalg.norm(w))
    return (w / n if n > 0 else w), n
