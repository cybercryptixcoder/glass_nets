"""D1: give the discovery-set networks readable catalogue names (meta.json name/family/answer). Display only: the
decompiler reads weights.npz and nothing else."""
import glob, json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAMES = {"secret_add": "a+b mod 113 under a secret relabelling", "xyxy": "a+b+ab mod 113", "secret_z8z14": "the group Z8×Z14 under a secret labelling",
         "secret_s4": "the group S4 under a secret labelling", "sqsum": "a²+b² mod 113", "pow": "a^b mod p"}
for d in sorted(glob.glob(os.path.join(ROOT, "lab", "models", "dc_*"))):
    tj, mj = os.path.join(d, "train.json"), os.path.join(d, "meta.json")
    if not (os.path.exists(tj) and os.path.exists(mj)):
        continue
    t = json.load(open(tj)); m = json.load(open(mj))
    task, seed, p = t["task"], t["seed"], t.get("p", 113)
    nm = NAMES.get(task, task).replace("mod p", f"mod {p}")
    m["name"] = f"{nm} (seed {seed})"
    m["family"] = "Decompiler discovery set (trained here)"
    m["answer"] = {"short": t["answer_key"]["rule"], "detail": f"Trained on {int(t['frac'] * 100)}% of all pairs; accuracy on all pairs {t['all_acc']:.4f}. The decompiler is not shown this."}
    m["params"] = int(sum(__import__('numpy').prod(v.shape) for v in __import__('numpy').load(os.path.join(d, "weights.npz")).values()))
    json.dump(m, open(mj, "w"), indent=1)
    print(os.path.basename(d), "->", m["name"])
