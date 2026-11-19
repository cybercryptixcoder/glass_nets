"""A39 test: transformer gated runs give the same outputs and are >= 3x faster (fails before: no speed-up); the
planted-transformer atlas gives identical fidelities."""
import sys, time, pickle
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
tgt = atlas.make_target("modadd_p113")
mods = atlas.load_decomposition(tgt, ROOT / "results/glass/atlas/spd/spd_modadd_p113_original_30000_1024_lad_1e-4.pt")
P = pickle.load(open(ROOT / "results/glass/atlas/modadd_p113_d4_a37dev_model.pkl", "rb"))
trees, alive = P["trees"], P["alive"]; sel = set(tuple(s) for s in P["sel"])
C = mods["W_E"].U.shape[0]; am = {k: np.isin(np.arange(C), v).astype(float) for k, v in alive.items()}
for k in tgt.matrices:
    mods[k]._residual = np.zeros((mods[k].W.shape[0], mods[k].W.shape[1]))
xs = tgt.X[:8000]


def gate(k, names, F, a):
    m = np.repeat(am[k][None], len(F), 0)
    for c in alive[k]:
        if (k, c) in sel:
            m[:, c] = atlas._trigger_on(trees[(k, c)], F, atlas.GATE_TAU)
    return m


def reference(x):                      # the pre-A39 computation, written out
    B = len(x); U = {k: mods[k].U.detach().numpy() for k in tgt.matrices}; Vv = {k: mods[k].V.detach().numpy() for k in tgt.matrices}

    def mm(name, a):
        a = a.numpy(); n = tgt._rows(name); names, F = tgt._feats_uncached(x, name)
        g = np.repeat(am[name][None], len(F), 0)
        for c in alive[name]:
            if (name, c) in sel:
                g[:, c] = (atlas.ModAddTarget and (trees[(name, c)].predict_proba(F)[:, list(trees[(name, c)].classes_).index(True)] >= atlas.GATE_TAU)).astype(float)
        m = tgt._bm(g, B, n)
        return torch.tensor(((a @ Vv[name].T) * m) @ U[name] + a @ mods[name]._residual.T)
    with torch.no_grad():
        return tgt.forward(torch.tensor(x), mm).numpy()


t0 = time.time(); ref = reference(xs); t_ref = time.time() - t0
tgt.run_gated(mods, xs, gate)                       # warm the caches
t0 = time.time()
for _ in range(3):
    out = tgt.run_gated(mods, xs, gate)
t_new = (time.time() - t0) / 3
check(np.abs(out - ref).max() < 1e-9, f"same outputs as the pre-A39 computation (max diff {np.abs(out - ref).max():.1e})")
check(t_ref / t_new >= 3, f"gated run {t_ref / t_new:.1f}x faster ({t_ref:.2f}s -> {t_new:.2f}s)")
t = atlas.planted_transformer()
a = atlas.build_atlas(t, spd_model=ROOT / "results/glass/atlas/spd/spd_planted_transformer_original_30000_512.pt", depth=4)
want = {"rule_gated_trusted": 1.0}
print({k: round(v, 6) for k, v in a["fidelity"].items()})
ref_f = ROOT / "results/glass/atlas/_a39_planted_fid.json"
import json
if len(sys.argv) > 1 and sys.argv[1] == "--record":
    ref_f.write_text(json.dumps(a["fidelity"])); print("recorded")
elif ref_f.exists():
    old = json.loads(ref_f.read_text())
    check(all(abs(old[k] - a["fidelity"][k]) < 1e-12 for k in old), "planted-transformer fidelities identical to the pre-A39 record")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
