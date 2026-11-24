"""Concept atlas: which known concepts are linearly readable at which site, against controls.

Protocol (from the 2026 video-physics probing literature and McGrath et al. 2022): ridge regression on standardized,
PCA-reduced site activations; train/test split by group (game or trajectory) so that no test group is seen in
fitting; the penalty is chosen on training data only (leave-one-out ridge); controls: the same probe on an untrained
copy of the network, and a shuffled-target null.
"""
import numpy as np
import torch

ALPHAS = np.logspace(-2, 5, 15)


def pca_basis(X, k, seed=0):
    """Top-k principal directions of X [n, d] (float32). Returns (mean [d], basis [d, k])."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    mu = Xt.mean(0)
    k = min(k, Xt.shape[0] - 1, Xt.shape[1])
    U, S, V = torch.pca_lowrank(Xt - mu, q=k, center=False, niter=3)
    return mu.numpy(), V[:, :k].numpy()


def project(X, mu, V):
    return (X.astype(np.float32) - mu) @ V


class Ridge:
    """Ridge with the penalty chosen by exact leave-one-out error on the training rows (SVD form)."""

    def fit(self, Z, y):
        self.zm, self.zs = Z.mean(0), Z.std(0) + 1e-8
        self.ym = y.mean()
        A = (Z - self.zm) / self.zs
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Uy = U.T @ (y - self.ym)
        best = None
        for a in ALPHAS:
            d = s / (s ** 2 + a)
            coef = Vt.T @ (d * Uy)
            h = np.sum(U ** 2 * (s ** 2 / (s ** 2 + a)), axis=1)
            res = (y - self.ym) - A @ coef
            loo = np.mean((res / np.maximum(1 - h, 1e-6)) ** 2)
            if best is None or loo < best[0]:
                best = (loo, a, coef)
        self.alpha, self.coef = best[1], best[2]
        return self

    def predict(self, Z):
        return ((Z - self.zm) / self.zs) @ self.coef + self.ym

    def direction(self):
        """The probe's direction in the (unstandardized) input space."""
        return self.coef / self.zs


def r2(y, yhat):
    v = np.var(y)
    return float(1 - np.mean((y - yhat) ** 2) / v) if v > 0 else float("nan")


def auc(y, s):
    """Area under the ROC curve for binary y (ties averaged)."""
    y = np.asarray(y) > 0.5
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def is_binary(y):
    u = np.unique(y)
    return len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}


def probe_scores(Z_tr, y_tr, Z_te, y_te):
    """Fit on train, score on test: R^2 (and AUC when the concept is yes/no)."""
    if np.var(y_tr) == 0:
        return {"r2": float("nan"), "auc": float("nan")}
    m = Ridge().fit(Z_tr, y_tr)
    p = m.predict(Z_te)
    out = {"r2": r2(y_te, p), "alpha": float(m.alpha)}
    out["auc"] = auc(y_te, p) if is_binary(y_te) else float("nan")
    return out


def group_split(groups, test_frac=0.25, seed=0):
    """Indices for train/test with every group entirely on one side."""
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    test_g = set(rng.choice(g, int(len(g) * test_frac), replace=False).tolist())
    te = np.array([i for i, x in enumerate(groups) if x in test_g])
    tr = np.array([i for i, x in enumerate(groups) if x not in test_g])
    return tr, te
