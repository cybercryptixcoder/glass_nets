"""A41 test (written before the code): a mechanism whose importance is a half-circle at one frequency of a carries over to
unseen token values (label 'rule'); one whose importance is a random set of token values is predictable on held-out
inputs (trusted) but does not carry over (label 'lookup'). Fails before A41 (no atlas.value_heldout / atlas.mech_label)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
from sklearn.tree import DecisionTreeClassifier
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)


if not hasattr(atlas, "value_heldout") or not hasattr(atlas, "mech_label"):
    check(False, "atlas has value_heldout and mech_label"); print("SOME FAILED"); sys.exit(1)
tgt = atlas.make_target("modadd_p113")
p = tgt.p; X = tgt.X[:16000]; half = len(X) // 2; xtr, xte = X[:half], X[half:]
names, Ftr = tgt._feats_uncached(xtr, "W_in"); _, Fte = tgt._feats_uncached(xte, "W_in")
ttr, tte = tgt.row_tokens(xtr, "W_in"), tgt.row_tokens(xte, "W_in")
atr, ate = xtr.argmax(-1)[:, 0], xte.argmax(-1)[:, 0]


def run(fn, label):
    ytr, yte = fn(atr), fn(ate)
    tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Ftr, ytr)
    pr = tree.predict(Fte); conf = 0.5 * (pr[yte].mean() + (~pr[~yte]).mean())
    vh = atlas.value_heldout(Ftr, ytr, ttr, Fte, yte, np.ones(len(yte), bool), tte, p, depth=4)
    lab = atlas.mech_label(conf, vh)
    print(f"  {label}: held-out inputs {conf:.3f}, unseen values {vh}, label {lab}")
    return conf, vh, lab


conf, vh, lab = run(lambda a: np.cos(2 * np.pi * 8 * a / p + 1.0) > 0.3, "half-circle at frequency 8")
check(conf >= 0.9 and lab == "rule", "a half-circle mechanism is labelled rule")
S = set(np.random.default_rng(5).choice(p, 8, replace=False).tolist())
conf, vh, lab = run(lambda a: np.isin(a, list(S)), "8 random token values")
check(conf >= 0.9, "the random-set mechanism is trusted on held-out inputs (precondition: the rule lists the cases it saw)")
check(lab == "lookup", "the random-set mechanism is labelled lookup")
check(atlas.mech_label(0.5, 0.95) == "not explained", "an untrusted mechanism is 'not explained' whatever its unseen-value score")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
