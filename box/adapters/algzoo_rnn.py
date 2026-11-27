"""Adapter for AlgZoo's tiny ReLU RNNs (lab/models/zoo_2nd_argmax_*): run, and intervene at a named site.

h_0 = 0; h_t = relu(x_t * W_x + h_(t-1) @ W_h); out = h_T @ W_o; the decision is argmax(out).
Sites: h1 .. hT (the hidden state after each step) and out. A part's location is (site, unit direction u, scale s):
the network's value of the part is s * (h_site . u), one of SPEC R3's allowed linear maps (a direction, with a
declared scale).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AlgZooRNN:
    def __init__(self, target, weights=None):
        W = weights if weights is not None else dict(np.load(os.path.join(ROOT, "lab", "models", target, "weights.npz")))
        self.Wx, self.Wh, self.Wo = (np.asarray(W[k], np.float64) for k in ("W_x", "W_h", "W_o"))
        self.T = int(np.asarray(W.get("seq_len", [self.Wx.shape[0]])).ravel()[0]) if "seq_len" in W else None
        self.hidden = self.Wh.shape[0]
        self.target = target

    def sites_for(self, T):
        return [f"h{t}" for t in range(1, T + 1)] + ["out"]

    def run(self, X, edits=None, return_sites=False):
        edits = edits or {}
        X = np.asarray(X, np.float64)
        h = np.zeros((len(X), self.hidden))
        site = {}
        for t in range(X.shape[1]):
            h = np.maximum(X[:, t:t + 1] * self.Wx + h @ self.Wh, 0.0)
            name = f"h{t + 1}"
            if name in edits:
                h = edits[name](h)
            site[name] = h
        out = h @ self.Wo
        if "out" in edits:
            out = edits["out"](out)
        site["out"] = out
        return (out, site) if return_sites else out

    def outputs(self, X, edits=None):
        return self.run(X, edits).argmax(1)
