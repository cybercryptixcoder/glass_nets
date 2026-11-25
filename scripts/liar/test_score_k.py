"""Known cases for score_k.py, run before it touches real data.
Must pass: planted separation on probe 0 for both models -> K1 pass, AUC near 1.
Must fail: the same data with labels shuffled within each model -> all AUCs near 0.5, K1 fails.
Must flag K3: the planted signal is a shift of every probe (a yes-bias) -> mean-feature AUC equals the detector's."""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def make(d, tag, mode, seed):
    rng = np.random.default_rng(seed); rows = []; F = []
    for qi in range(744):
        for cond in ("lie", "truth"):
            lab = cond if rng.random() > 0.15 or cond == "truth" else "failed_lie"
            x = rng.normal(size=65)
            sig = 1.0 if lab == "lie" else 0.0
            if mode == "probe0": x[0] += 3 * sig
            if mode == "shift": x += 1.0 * sig
            F.append(x); rows.append(dict(qi=qi, cond=cond, label=lab))
    F = np.array(F, dtype=np.float32)
    if mode == "shuffled":
        lab = np.array([r["label"] for r in rows]); rng.shuffle(lab)
        for r, l in zip(rows, lab): r["label"] = str(l)
    np.save(os.path.join(d, f"{tag}_feats.npy"), F); np.save(os.path.join(d, f"{tag}_feats_noinstr.npy"), F)
    with open(os.path.join(d, f"{tag}_transcripts.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def run(mode):
    d = tempfile.mkdtemp()
    src = mode if mode != "shuffled" else "probe0"
    make(d, "a", src, 1); make(d, "b", src, 2)
    if mode == "shuffled":
        make(d, "a", "shuffled", 1); make(d, "b", "shuffled", 2)
    out = os.path.join(d, "r.json")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "score_k.py"), d, "a", "b", out], check=True, capture_output=True)
    return json.load(open(out))
r = run("probe0"); a = r["auc"]["train a -> test b"]["lie_vs_truth"]
assert r["K1_pass"] and a > 0.95 and not r["K3_yes_bias"], r
r = run("shuffled"); a = r["auc"]["train a -> test b"]["lie_vs_truth"]
assert (not r["K1_pass"]) and abs(a - 0.5) < 0.06, a
r = run("shift"); assert r["K3_yes_bias"], r["auc"]["train a -> test b"]
print("score_k known cases: pass")
