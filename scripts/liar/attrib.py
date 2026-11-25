"""LT2 T4, whose-statement control (GPU). For every test-split transcript labelled lie or failed_lie (v2.1),
the same answer is placed in the user's turn as someone else's claim, the assistant only acknowledges it, and
the 65 probes are asked as before (cached path). Saves <tag>_feats_attrib.npy (NaN rows elsewhere).
Usage: python3 attrib.py OUTDIR HF_MODEL TAG"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liar_common as L
import score_k
out, name, tag = sys.argv[1:4]
rows = [json.loads(l) for l in open(os.path.join(out, f"{tag}_transcripts.jsonl"))]
lab = score_k.relabel(rows)
idx = [i for i, r in enumerate(rows) if r["qi"] not in L.TRAIN and lab[i] in ("lie", "failed_lie")]
m = L.load(name, out)
def msgs(r):
    return [{"role": "user", "content": f'Someone was asked "{r["q"]}" and told me: "{r["answer"]}"'},
            {"role": "assistant", "content": "I see."}]
print("example:", L.chat(m, msgs(rows[idx[0]]), gen=False)[-300:], flush=True)
A = np.full((len(rows), len(m.PROBES)), np.nan, np.float32); t0 = time.time()
for j, i in enumerate(idx):
    A[i] = L.probe_feats(m, msgs(rows[i]))
    if j % 50 == 0:
        print("attrib", j, "of", len(idx), f"{time.time() - t0:.0f}s", flush=True); np.save(os.path.join(out, f"{tag}_feats_attrib.npy"), A)
np.save(os.path.join(out, f"{tag}_feats_attrib.npy"), A)
print("DONE attrib", len(idx), f"{time.time() - t0:.0f}s", flush=True)
