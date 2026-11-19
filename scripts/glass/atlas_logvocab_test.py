"""A40 test (written before the change). Run with ATLAS_MODADD_VOCAB=both.
(a) modmul_p113's result feature is (a·b) mod 113 and its discrete-log angles are right; (b) a half-circle in the log
basis is fit with only that log-basis feature; (c) under 'both', a number-line half-circle on modadd still picks the
number-line feature. Fails before A40 (no operation, no log features)."""
import os, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
os.environ.setdefault("ATLAS_MODADD_VOCAB", "both")
import atlas
from sklearn.tree import DecisionTreeClassifier
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)


def fit(names_, F_, y):
    n = len(y) // 2
    tr = DecisionTreeClassifier(max_depth=4, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(F_[:n], y[:n])
    pr = tr.predict(F_[n:]); yt = y[n:]
    atlas.rule_text(tr, names_)
    return 0.5 * (pr[yt].mean() + (~pr[~yt]).mean()), tr._features_used


p = 113
mul = atlas.make_target("modmul_p113")
x = mul.X[:6000]; t = x.argmax(-1); a, b = t[:, 0], t[:, 1]
names, F = mul._feats_uncached(x, "W_in")
check(f"(a·b) mod {p}" in names and np.array_equal(F[:, names.index(f"(a·b) mod {p}")], (a * b) % p), "modmul's result feature is (a·b) mod 113")
check(not any("(a+b)" in n for n in names), "no (a+b) feature for a multiplication network")
la = F[:, names.index("log a")]
nz = a != 0
check(np.all(la[~nz] == -1) and np.all(np.array([pow(3, int(e), p) for e in la[nz]]) == a[nz]), "log a: 3^(log a) = a, and log 0 = -1")
nm = f"4·log a mod {p - 1}"
check(nm in names and np.array_equal(F[nz, names.index(nm)], (4 * la[nz]) % (p - 1)) and np.all(np.isnan(F[~nz, names.index(nm)])),
      f"'{nm}' has the right values (undefined at a = 0)")
check(f"8·a mod {p}" in names, "number-line angles are present too")
y = np.where(nz, np.cos(2 * np.pi * 4 * np.where(nz, la, 0) / (p - 1) + 0.7) > 0.2, False)
acc, used = fit(names, F, y)
check(acc >= 0.97 and all("log" in u for u in used) and {int(m) for u in used for m in re.findall(r"(\d+)·", u)} <= {4},
      f"log-basis half-circle: accuracy {acc:.3f}, features {used}")
add = atlas.make_target("modadd_p113")
xa = add.X[:6000]; ta = xa.argmax(-1)
na, Fa = add._feats_uncached(xa, "W_in")
ya = np.cos(2 * np.pi * 8 * ta[:, 0] / p + 1.0) > 0.3
acc, used = fit(na, Fa, ya)
check(acc >= 0.97 and used == [f"8·a mod {p}"], f"number-line half-circle on modadd: accuracy {acc:.3f}, features {used}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
