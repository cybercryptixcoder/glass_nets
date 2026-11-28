"""boxkit: the toolkit an evolved box may import (the only non-standard module it may import).

It gives the box the network it explains (a `Net`), the frozen starting vocabulary for numeric sequences (domain
`seq`, bench/accept/VOCABULARY.md), and a few search and fitting helpers. It knows nothing about how the box is
scored: the evaluator is a separate, sealed program.

A Net (a one-layer ReLU recurrent network, as in ARC's AlgZoo):
    h_0 = 0;  h_t = relu(x_t * W_x + h_(t-1) @ W_h)  for t = 1..T;  out = h_T @ W_o;  decision = argmax(out)
    net.T, net.hidden, net.n_out, net.task (one line, or "unknown")
    net.weights: {"W_x": [1, H], "W_h": [H, H], "W_o": [H, n_out]} (copies; changing them changes nothing)
    net.sites: ["h1", ..., "hT", "out"]
    net.inputs(n, seed): n input sequences [n, T] from the network's input distribution
    net.run(X, edits=None) -> (out [n, n_out], sites {"h1": [n, H], ..., "out": [n, n_out]})
    net.decide(X, edits=None) -> decisions [n] (integers)
    edits: {site: function(h [n, H]) -> h [n, H]}, applied to that site's activations as the network runs
           (an intervention: later steps see the edited state)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from bench.accept import vocab as _V  # noqa: E402  (the frozen vocabulary)
from box import search as _S  # noqa: E402

DOMAIN = "seq"
L = 7                 # longest definition, in symbols (part references and invented primitives count as one)
MAX_CONSTANTS = 5     # constants per definition, after expanding invented primitives
SIMPLE = sorted({k / 2 for k in range(-20, 21)})   # constants the evaluator counts as simple: integers and halves in [-10, 10]
_DOM = _V.get(DOMAIN)


class Net:
    def __init__(self, W, T, task="unknown", input_kind="normal"):
        self._Wx = np.asarray(W["W_x"], float).reshape(1, -1)
        self._Wh = np.asarray(W["W_h"], float)
        self._Wo = np.asarray(W["W_o"], float)
        self.T, self.hidden, self.n_out = int(T), self._Wh.shape[0], self._Wo.shape[1]
        self.task = task
        self._kind = input_kind
        self.sites = [f"h{t}" for t in range(1, self.T + 1)] + ["out"]

    @property
    def weights(self):
        return {"W_x": self._Wx.copy(), "W_h": self._Wh.copy(), "W_o": self._Wo.copy()}

    def inputs(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((int(n), self.T))

    def run(self, X, edits=None):
        edits = edits or {}
        X = np.asarray(X, float)
        h = np.zeros((len(X), self.hidden))
        sites = {}
        for t in range(self.T):
            h = np.maximum(X[:, t:t + 1] * self._Wx + h @ self._Wh, 0.0)
            name = f"h{t + 1}"
            if name in edits:
                h = np.asarray(edits[name](h), float)
            sites[name] = h
        out = h @ self._Wo
        if "out" in edits:
            out = np.asarray(edits["out"](out), float)
        sites["out"] = out
        return out, sites

    def decide(self, X, edits=None):
        return self.run(X, edits)[0].argmax(1)


# ---------------------------------------------------------------- the vocabulary
def evaluate(expr, X, parts=None, types=None, library=None):
    """The value of a definition on inputs X [n, T]: a Num gives [n], a Seq gives [n, T]. parts: {id: values},
    types: {id: "Num" or "Seq"}, library: invented primitives (see `admissible`)."""
    v = _V.evaluate(expr, _DOM, {"x": np.asarray(X, float)}, parts or {}, types or {}, library or {})
    return np.asarray(v, float)


def admissible(expr, types=None, library=None):
    """(ok, why, type): whether a definition is a well-formed expression of the vocabulary within L and the
    constant budget, given the parts it refers to ({id: type}) and the invented primitives (library)."""
    return _V.admissible(expr, _DOM, types or {}, library or {})


def enumerate_exprs(X, max_len, parts=None, cap=60000):
    """Every distinct expression up to max_len symbols, as {type: [(text, values on X, length)]} (distinct by value
    on X: use a few dozen probe inputs). parts: {id: (type, values on X)} lets expressions refer to earlier parts."""
    return _S.enumerate_exprs(_DOM, {"x": np.asarray(X, float)}, max_len, cap=cap, parts=parts)


def best_definitions(X_fit, target_fit, max_len, root="Num", k=5, X_test=None, target_test=None, parts_fit=None,
                     parts_test=None, cap=60000):
    """The k expressions of type root (up to max_len symbols) best correlated with target on X_fit, each re-scored on
    X_test when given: a list of dicts with "definition", "corr" (on the fit inputs), "heldout_corr", "scale" and
    "offset" (the affine fit). parts_*: {id: (type, values)}."""
    res, _ = _S.best_definitions(_DOM, {"x": np.asarray(X_fit, float)}, np.asarray(target_fit, float), max_len, root,
                                 k=k, ctx_test=None if X_test is None else {"x": np.asarray(X_test, float)},
                                 target_test=target_test, cap=cap, parts_fit=parts_fit, parts_test=parts_test)
    return res


def fit_location(sites, site, values, idx):
    """A unit direction u and a scale s with s * (sites[site] @ u) ~ values, fitted by least squares on rows idx
    (one unit alone when it reproduces the values almost exactly)."""
    from box.verify_vec import fit_location as _fit
    u, s = _fit(sites, site, np.asarray(values, float), np.asarray(idx))
    return np.asarray(u, float), float(s)


def is_simple(c):
    return any(abs(float(c) - s) < 1e-9 for s in SIMPLE)
