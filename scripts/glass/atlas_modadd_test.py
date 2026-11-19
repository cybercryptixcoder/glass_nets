"""Test for CHANGES A35 (ModAddTarget). Fails before (no adapter), passes after.
Checks the adapter, not the atlas: forward equals the reference; call rows and features have the right shapes; the
domain halves are balanced. The Fourier known answer is judged on the atlas once its decomposition exists."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas, judge, nets
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
try:
    t = atlas.make_target("modadd_p113")
except Exception as e:  # noqa: BLE001
    check(False, f"make_target('modadd_p113') works ({type(e).__name__}: {e})"); print(f"{len(fails)} FAILED"); sys.exit(1)
check(getattr(t, "kind", "") == "transformer", "kind is transformer")
meta, W, X = judge.load_target("modadd_p113", "original", 0)
x = t.X[:512]; toks = x.argmax(-1)
ref = nets.onelayer_transformer(W, toks)
with torch.no_grad():
    out = t.forward(torch.tensor(x), lambda n, a: a @ t.weights[n].T).numpy() * t.out_scale
check(np.abs(out - ref).max() < 1e-9, f"forward equals nets.onelayer_transformer (max error {np.abs(out - ref).max():.2e})")
mods = torch.nn.ModuleDict({k: atlas.spd.Decomposed(v, 4).double() for k, v in t.weights.items()})
calls = t.network_calls(mods, x[:64])
for k in t.matrices:
    names, F, G = calls[k]; n = t._rows(k)
    check(F.shape == (64 * n, len(names)) and G.shape == (64 * n, 4), f"{k}: {n} call(s) per input, features {F.shape}, importances {G.shape}")
names, F, _ = calls["W_U"]
i = names.index("(a+b) mod 113"); a_ = names.index("a"); b_ = names.index("b")
check(np.all(F[:, i] == (F[:, a_] + F[:, b_]) % 113), "feature (a+b) mod 113 is right")
h = len(t.X) // 2; A = t.X.argmax(-1)[:, 0]
check(abs(A[:h].mean() - A[h:].mean()) < 2.0, f"domain halves balanced (mean a {A[:h].mean():.1f} vs {A[h:].mean():.1f})")
with torch.no_grad():
    g = t.run_gated(mods, x[:64], lambda k, n_, F_, a_: np.ones((len(F_), 4)))
    full = t.forward(torch.tensor(x[:64]), lambda n_, a_: mods[n_].apply(a_, torch.ones(mods[n_].U.shape[0], dtype=a_.dtype))).numpy()   # the decomposition's sum
check(np.abs(g - full).max() < 1e-9, "run_gated with every mechanism on equals the decomposed forward")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
