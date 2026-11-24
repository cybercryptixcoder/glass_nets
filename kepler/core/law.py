"""Law extraction: which few terms of a candidate library reproduce a quantity, with coefficients.

Used on a MODEL's implied dynamics (not on data): y = the model's implied time derivative, Theta = candidate terms.
Selection rule (fixed before any model result, K1): least squares on the training rows, then backward elimination on
held-out rows. A term is dropped while dropping it raises held-out MSE by at most EPS of the current held-out MSE
(the unexplained error); the first term whose removal costs more than that stops the elimination. A variance cutoff
is avoided on purpose: it would drop small true terms (drag) whose share of the total variance is tiny but whose
share of the unexplained error is not.
"""
import numpy as np

EPS = 0.01


def _ols(A, y):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _mse(A, y, coef):
    r = y - A @ coef
    return float(np.mean(r * r))


def r2(y, yhat):
    return 1.0 - float(np.mean((y - yhat) ** 2) / np.var(y))


def extract_law(Theta_tr, y_tr, Theta_va, y_va, names, eps=EPS, keep=()):
    """Backward elimination. Returns dict(terms, coefs, r2_val, path). `keep` names are never dropped.
    Fits use Gram matrices (exactly the least-squares solutions on the rows, computed from 38 x 38 systems)."""
    scale = np.sqrt(np.mean(Theta_tr ** 2, axis=0)) + 1e-30
    A_tr, A_va = Theta_tr / scale, Theta_va / scale
    G_tr, b_tr = A_tr.T @ A_tr, A_tr.T @ y_tr
    G_va, b_va, yy_va, n_va = A_va.T @ A_va, A_va.T @ y_va, float(y_va @ y_va), len(y_va)

    def fit(idx):
        idx = np.array(idx)
        return np.linalg.lstsq(G_tr[np.ix_(idx, idx)], b_tr[idx], rcond=None)[0]

    def mse_va(idx, c):
        idx = np.array(idx)
        return (yy_va - 2 * c @ b_va[idx] + c @ G_va[np.ix_(idx, idx)] @ c) / n_va

    active = list(range(len(names)))
    cur = mse_va(active, fit(active))
    path = [{"n_terms": len(active), "mse_val": float(cur), "dropped": None}]
    while len(active) > 1:
        best = None
        for j in active:
            if names[j] in keep:
                continue
            trial = [k for k in active if k != j]
            m = mse_va(trial, fit(trial))
            if best is None or m < best[1]:
                best = (j, m)
        # numerical floor: an unexplained error below 1e-9 of the target variance is round-off, not signal
        if best is None or best[1] > max(cur, 1e-9 * yy_va / n_va) * (1 + eps):
            break
        active = [k for k in active if k != best[0]]
        cur = best[1]
        path.append({"n_terms": len(active), "mse_val": float(cur), "dropped": names[best[0]]})
    coef = fit(active)
    yhat = A_va[:, active] @ coef
    return {
        "terms": [names[k] for k in active],
        "coefs": {names[k]: float(c / scale[k]) for k, c in zip(active, coef)},
        "r2_val": r2(y_va, yhat),
        "mse_val": float(np.mean((y_va - yhat) ** 2)),
        "path": path,
    }


def term_importance(Theta_va, y_va, law):
    """Share of the explained variance each kept term carries (drop-one increase of held-out MSE)."""
    names = law["terms"]
    cols = {n: i for i, n in enumerate(names)}
    full = np.stack([Theta_va[:, i] for i in range(Theta_va.shape[1])], 1)
    coefs = np.array([law["coefs"][n] for n in names])
    base = float(np.mean((y_va - full @ coefs) ** 2))
    out = {}
    for n in names:
        c = coefs.copy()
        c[cols[n]] = 0.0
        out[n] = float(np.mean((y_va - full @ c) ** 2) - base) / float(np.var(y_va))
    return out
