"""What known concepts explain of a network's output, and what they leave unexplained.

surrogate: gradient-boosted trees from the known concept library to the network's output (deterministic settings).
gain test: does a candidate concept, added to the library, raise the surrogate's held-out R^2? Paired bootstrap over
test rows gives the 95% interval of the gain.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def surrogate(X_tr, y_tr, seed=0):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                      l2_regularization=1.0, early_stopping=False, random_state=seed)
    return m.fit(X_tr, y_tr)


def r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def gain(X_tr, y_tr, X_te, y_te, new_tr, new_te, n_boot=1000, seed=0, base_pred=None):
    """Held-out R^2 gain from adding columns new_* to the library X_*; returns point and 95% bootstrap interval."""
    if base_pred is None:
        base_pred = surrogate(X_tr, y_tr).predict(X_te)
    aug = surrogate(np.column_stack([X_tr, new_tr]), y_tr).predict(np.column_stack([X_te, new_te]))
    e0, e1 = (y_te - base_pred) ** 2, (y_te - aug) ** 2
    rng = np.random.default_rng(seed)
    n = len(y_te)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = np.var(y_te[i])
        boots.append((e0[i].mean() - e1[i].mean()) / v)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"gain": float((e0.mean() - e1.mean()) / np.var(y_te)), "ci95": [float(lo), float(hi)],
            "r2_base": r2(y_te, base_pred), "r2_aug": r2(y_te, aug)}, aug
