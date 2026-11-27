"""The checker, v0 (GAP.md C2; ledger row G1): certify a blueprint or reject an account, with reasons.

What v0 checks (SPEC.md R3, R4, R5, R9), in order, stopping at the first failure that rules the account out:
1. Form: the account is a Blueprint; its wiring has no cycle; every location is an allowed linear map (a subspace of
   a named site, by basis labels or an orthonormal basis); every rule's definition is admissible in the starting
   vocabulary (grammar, length L, 5 constants after expansion) and refers only to its declared upstream parts.
2. Run alone (R4, SPEC §2 condition 3): every definition standing in at once, the blueprint reproduces the network's
   outputs. Agreement is reported and compared with a line the caller supplies (on the bench: the true blueprint's
   own score minus the tier's spread).
3. Interchange interventions (R3): for each located part, on pairs of inputs where swapping that part's value is
   predicted by the blueprint to change the output, the network with the part's subspace swapped must give the
   output the blueprint predicts. Scored against a family-wise line from random-subspace controls at the same site:
   the 99th percentile, over repeated draws, of the best of K control scores, K = the number of parts tested.
4. Substitution (R3): putting the blueprint's own value of the part into its subspace leaves the network's output
   unchanged, as a share of the change that resampling the part causes. Reported in v0; its line comes from the
   bench (the true blueprint's own score minus the tier's spread), which does not exist yet.
5. Typing (R5): a part typed lookup whose interventions on fresh inputs work above the control line is rejected (its
   causal effect carries over, so it is not a table).

Nothing here chooses a line by looking at the results it judges: the run-alone line is passed in, and the
intervention line comes from controls.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "accept"))
import vocab  # noqa: E402

from box.blueprint import Blueprint, TYPES  # noqa: E402


def _same(a, b, numerical, tol=1e-3):
    if numerical:
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))
    return list(a) == list(b)


class Checker:
    def __init__(self, net, inputs, seed=0, n_pairs=200, control_reps=200):
        self.net, self.inputs, self.rng = net, inputs, np.random.default_rng(seed)
        self.n_pairs, self.control_reps = n_pairs, control_reps
        self.base = net.outputs(inputs)
        _, self.sites, _ = net.run(inputs, return_sites=True)
        n = len(inputs)
        pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
        self.rng.shuffle(pairs)
        self.pairs = pairs[:n_pairs]

    # ------------------------------------------------------------------ 1. form
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
                return False, f"part {p.id}: type {p.type!r} is not one of {TYPES}"
            if p.site not in self.net.sites:
                return False, f"part {p.id}: unknown site {p.site}"
            if p.labels is None and p.basis is None:
                return False, f"part {p.id}: no location"
            if p.labels is not None and not all(l in self.net.labels for l in p.labels):
                return False, f"part {p.id}: location labels not in the network's residual basis"
            if p.basis is not None:
                B = np.asarray(p.basis, float)
                if B.ndim != 2 or B.shape[0] != self.net.d or not np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-6):
                    return False, f"part {p.id}: basis is not an orthonormal [d, k] matrix (allowed maps are linear)"
            if p.type == "rule":
                if not isinstance(p.definition, str):
                    return False, f"part {p.id}: a rule needs a vocabulary definition"
                ok, why, t = vocab.admissible(p.definition, dom, {u: types[u] for u in p.upstream}, bp.library)
                if not ok:
                    return False, f"part {p.id}: {why}"
                refs = {tok[1:] for tok in vocab.unparse(vocab.parse(p.definition)).replace("(", " ").replace(")", " ").split()
                        if tok.startswith("@")}
                if not refs <= set(p.upstream):
                    return False, f"part {p.id}: refers to parts not declared upstream: {sorted(refs - set(p.upstream))}"
                if t != p.vtype:
                    return False, f"part {p.id}: definition has type {t}, declared {p.vtype}"
            if p.type == "lookup" and not isinstance(p.table, dict):
                return False, f"part {p.id}: a lookup must list its table"
            types[p.id] = p.vtype
        if bp.output not in types:
            return False, f"output part {bp.output} not defined"
        return True, "ok"

    # ------------------------------------------------------------------ evaluation of the blueprint alone
    def values(self, bp, idx, overrides=None):
        """Every part's value on the inputs idx, with optional overrides {part id: values on idx}."""
        overrides = overrides or {}
        dom = vocab.get(bp.domain)
        ctx = {"tokens": [self.inputs[i] for i in idx]}
        vals, types = {}, {}
        for p in bp.order():
            if p.id in overrides:
                vals[p.id] = overrides[p.id]
            elif p.type == "rule":
                vals[p.id] = vocab.evaluate(p.definition, dom, ctx, vals, types, bp.library)
            elif p.type == "lookup":
                vals[p.id] = [[p.table.get((tuple(self.inputs[i]), j)) for j in range(len(self.inputs[i]))] for i in idx]
            else:
                vals[p.id] = [[None] * len(self.inputs[i]) for i in idx]
            types[p.id] = p.vtype
        return vals

    def run_alone(self, bp):
        idx = list(range(len(self.inputs)))
        out = self.values(bp, idx)[bp.output]
        agree = [_same(o, b, self.net.numerical_output) for o, b in zip(out, self.base)]
        return float(np.mean(agree))

    # ------------------------------------------------------------------ 3. interchange interventions
    def _basis(self, p):
        return self.net.subspace(p.labels) if p.labels is not None else np.asarray(p.basis, float)

    def _decisive(self, bp, pid):
        """Pairs where the blueprint predicts that swapping pid's value changes the output, with the prediction."""
        out = []
        for a, b in self.pairs:
            va = self.values(bp, [a])
            vb = self.values(bp, [b])
            # the network swap covers the aligned positions of both sequences; beyond the shorter one, a keeps its own
            own, other = list(va[pid][0]), list(vb[pid][0])
            m = min(len(own), len(other))
            mixed = other[:m] + own[m:]
            swapped = self.values(bp, [a], overrides={pid: [mixed]})[bp.output][0]
            if not _same(swapped, va[bp.output][0], self.net.numerical_output):
                out.append((a, b, swapped))
        return out

    def _iia(self, B, site, decisive):
        """A hit is a change of the network's output to the predicted value (an output that ignores the swap scores
        nothing)."""
        hits = []
        for a, b, pred in decisive:
            edit = {site: _swap(B, self.sites[site][b:b + 1])}
            got = self.net.outputs([self.inputs[a]], edit)[0]
            changed = not _same(got, self.base[a], self.net.numerical_output)
            hits.append(changed and _same(got, pred, self.net.numerical_output))
        return float(np.mean(hits)) if hits else None

    def iia(self, bp, p):
        dec = self._decisive(bp, p.id)
        return self._iia(self._basis(p), p.site, dec), dec

    def control_line(self, bp, tested, decisive_by_part, K):
        """Family-wise: the 99th percentile over repetitions of the best of K random-subspace control scores."""
        best = []
        for _ in range(self.control_reps // 10):
            scores = []
            for _k in range(K):
                p = tested[self.rng.integers(len(tested))]
                dec = decisive_by_part[p.id]
                if not dec:
                    continue
                k = self._basis(p).shape[1]
                Q, _ = np.linalg.qr(self.rng.normal(size=(self.net.d, k)))
                sub = [dec[i] for i in self.rng.choice(len(dec), min(len(dec), 20), replace=False)]
                scores.append(self._iia(Q, p.site, sub))
            if scores:
                best.append(max(scores))
        return float(np.quantile(best, 0.99)) if best else 1.0

    # ------------------------------------------------------------------ 4. substitution
    def substitution(self, bp, p):
        """Share of the output change caused by resampling the part that putting its own value back restores."""
        B = self._basis(p)
        idx = list(range(len(self.inputs)))
        vals = self.values(bp, idx)[p.id]
        coords = np.zeros((len(idx), self.sites[p.site].shape[1], B.shape[1]))
        labels = p.labels or []
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                if v is None:
                    continue
                if p.encoding == "numerical":
                    coords[i, j + 1, 0] = float(v)
                else:
                    name = f"{labels[0].split(':')[0]}:{v}" if labels else None
                    if name in labels:
                        coords[i, j + 1, labels.index(name)] = 1.0
        # keep each input's own value at the BOS position (the blueprint gives no value there)
        coords[:, 0] = self.sites[p.site][:, 0] @ B
        sub = self.net.outputs(self.inputs, {p.site: _set(B, coords)})
        perm = self.rng.permutation(len(idx))
        abl = self.net.outputs(self.inputs, {p.site: _swap_rows(B, self.sites[p.site][perm])})
        same = lambda xs: np.mean([_same(x, y, self.net.numerical_output) for x, y in zip(xs, self.base)])  # noqa: E731
        s_sub, s_abl = same(sub), same(abl)
        return float((s_sub - s_abl) / (1 - s_abl)) if s_abl < 1 else None

    # ------------------------------------------------------------------ the verdict
    def certify(self, bp, run_alone_line):
        report = {"name": getattr(bp, "name", ""), "checks": []}
        ok, why = self.form(bp)
        report["checks"].append(("form", ok, why))
        if not ok:
            report["verdict"] = "rejected"
            return report
        ra = self.run_alone(bp)
        ok = ra >= run_alone_line
        report["checks"].append(("run alone", ok, f"agreement {ra:.3f} against line {run_alone_line:.3f}"))
        located = [p for p in bp.parts if p.type in ("rule", "lookup")]
        dec, iias = {}, {}
        for p in located:
            iias[p.id], dec[p.id] = self.iia(bp, p)
        line = self.control_line(bp, located, dec, K=max(1, len(located)))
        report["iia_line"] = line
        for p in located:
            s = iias[p.id]
            if s is None:
                report["checks"].append((f"interventions {p.id}", p.type != "rule",
                                         "no decisive pairs: swapping it never changes the predicted output, so its "
                                         "location is untested (type it irrelevant or unresolved, or add inputs)"))
                continue
            good = s > line
            if p.type == "rule":
                report["checks"].append((f"interventions {p.id}", good, f"IIA {s:.3f} against family-wise line {line:.3f}"))
                sub = self.substitution(bp, p)
                report["checks"].append((f"substitution {p.id}", True,
                                         f"reported (line from the bench): restores "
                                         f"{sub if sub is None else round(sub, 3)} of the resampling change"))
            else:  # lookup
                report["checks"].append((f"typing {p.id}", not good,
                                         f"typed lookup, but its interventions work on fresh inputs (IIA {s:.3f} > line {line:.3f})"
                                         if good else f"lookup: no causal effect above the line (IIA {s:.3f})"))
        report["verdict"] = "certified" if all(c[1] for c in report["checks"]) else "rejected"
        return report


def _swap(B, h_src):
    P = B @ B.T

    def fn(h):
        T = min(h.shape[1], h_src.shape[1])
        out = h.copy()
        out[:, :T] = h[:, :T] - h[:, :T] @ P + h_src[:, :T] @ P
        return out
    return fn


def _swap_rows(B, h_src):
    return _swap(B, h_src)


def _set(B, coords):
    P = B @ B.T

    def fn(h):
        T = min(h.shape[1], coords.shape[1])
        out = h.copy()
        out[:, :T] = h[:, :T] - h[:, :T] @ P + coords[:, :T] @ B.T
        return out
    return fn
