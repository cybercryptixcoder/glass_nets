"""Known cases for score_t.py before real data. Must pass: lies separable from failed lies on probe 0 ->
T1 pass, T2 pass. Surface control: with random answer words the TF-IDF AUC is near 0.5 (tell beyond wording);
with a word that marks every lie, the surface AUC is near 1 and T3 flags it."""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
W = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta", "iota", "kappa"]
def make(d, tag, seed, mark):
    rng = np.random.default_rng(seed); rows = []; F = []
    for qi in range(744):
        for cond in ("lie", "truth"):
            lab = "truth" if cond == "truth" else ("lie" if rng.random() < 0.5 else "failed_lie")
            x = rng.normal(size=65); x[0] += 3.0 * (lab == "lie")
            words = " ".join(rng.choice(W, 6)) + (" zzz" if (mark and lab == "lie") else "")
            F.append(x); rows.append(dict(qi=qi, cond=cond, label=lab, answer=words))
    F = np.array(F, dtype=np.float32)
    np.save(os.path.join(d, f"{tag}_feats.npy"), F); np.save(os.path.join(d, f"{tag}_feats_noinstr.npy"), F)
    with open(os.path.join(d, f"{tag}_transcripts.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def run(mark):
    d = tempfile.mkdtemp(); make(d, "a", 1, mark); make(d, "b", 2, mark); out = os.path.join(d, "t.json")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "score_t.py"), d, "a", "b", out], check=True, capture_output=True)
    return json.load(open(out))
r = run(False)
assert r["T1_pass"] and r["T2_pass"], r
assert abs(r["T3 surface a"]["auc"] - 0.5) < 0.08 and not r["T3_not_beyond_wording"]["a"], r
r = run(True)
assert r["T3 surface a"]["auc"] > 0.95 and r["T3_not_beyond_wording"]["a"], r
print("score_t known cases: pass")
