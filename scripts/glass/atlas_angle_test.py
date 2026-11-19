"""A37 test (written before the change): the modular-arithmetic vocabulary carries angle features 'k·v mod p', and a
mechanism whose importance is a half-circle at one frequency is fit exactly by the atlas's tree settings using only
that frequency. Fails before A37 (no angle features), passes after."""
import re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
from sklearn.tree import DecisionTreeClassifier

tgt = atlas.make_target("modadd_p113")
x = tgt.X[:6000]; t = x.argmax(-1); a, b = t[:, 0], t[:, 1]; p = tgt.p
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)


# (a) features
names, F = tgt._feats_uncached(x, "W_in")
check(not any(n.startswith(("cos(", "sin(")) for n in names), "no cos/sin features remain")
for v, val in (("a", a), ("b", b), ("(a+b)", (a + b) % p)):
    nm = f"8·{v} mod {p}"
    check(nm in names and np.array_equal(F[:, names.index(nm)], (8 * val) % p), f"W_in rows carry '{nm}' with the right values")
namesE, FE = tgt._feats_uncached(x, "W_E")
here = atlas.ModAddTarget._pm(tgt, t[:, :, None].astype(float), len(x), 3)[:, 0]
check(f"8·token here mod {p}" in namesE and np.array_equal(FE[:, namesE.index(f"8·token here mod {p}")], (8 * here) % p),
      "W_E rows carry '8·token here mod 113' with the right values")
check(len(names) == 5 + 4 * (p // 2 - 1), f"{len(names)} features (expected {5 + 4 * (p // 2 - 1)})")


# (b) a half-circle at frequency 8 is fit exactly, with frequency-8 features only
def fit(names_, F_, y):
    n = len(y) // 2
    tr = DecisionTreeClassifier(max_depth=4, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(F_[:n], y[:n])
    pr = tr.predict(F_[n:]); yt = y[n:]
    acc = 0.5 * (pr[yt].mean() + (~pr[~yt]).mean())
    atlas.rule_text(tr, names_)
    freqs = {int(f) for u in tr._features_used for f in re.findall(r"(\d+)·", u)}
    return acc, tr._features_used, freqs


y_in = np.cos(2 * np.pi * 8 * a / p + 1.0) > 0.3
acc, used, fr = fit(names, F, y_in)
check(acc >= 0.97 and fr == {8} and all(re.search(r"(\d+)·", u) for u in used), f"W_in half-circle in a: accuracy {acc:.3f}, features {used}")
y_e = np.cos(2 * np.pi * 8 * here / p + 1.0) > 0.3
acc, used, fr = fit(namesE, FE, y_e)
check(acc >= 0.97 and fr == {8}, f"W_E half-circle in the token: accuracy {acc:.3f}, features {used}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
