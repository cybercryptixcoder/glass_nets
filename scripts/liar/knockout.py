"""LT2-M4: probe features with the probe tokens' attention to the instruction tokens masked out (the transcript
itself is encoded normally). Test split, labels lie / failed_lie / truth. Saves <tag>_feats_knockout.npy.
Usage: python3 knockout.py OUTDIR HF_MODEL TAG [SPAN]   SPAN = instruction (M4, default), user or assistant (M5)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liar_common as L
import score_k
out, name, tag = sys.argv[1:4]
span = sys.argv[4] if len(sys.argv) > 4 else "instruction"
suffix = "" if span == "instruction" else "_" + span
rows = [json.loads(l) for l in open(os.path.join(out, f"{tag}_transcripts.jsonl"))]
F = np.load(os.path.join(out, f"{tag}_feats.npy"))
lab = score_k.relabel(rows)
idx = [i for i, r in enumerate(rows) if r["qi"] not in L.TRAIN and lab[i] in ("lie", "failed_lie", "truth")]
m = L.load(name, out)
def instr(r): return (L.LIE if r["cond"] == "lie" else L.TRUTH)[r["k"]]
b0 = L.base_msgs(rows[idx[0]])
u, e = np.array(L.probe_feats(m, b0)), np.array(L.probe_feats(m, b0, hide=[]))
assert np.abs(u - e).max() == 0.0, "empty mask changed the features"
hp = L.instruction_positions(m, b0, instr(rows[idx[0]]))
if span != "instruction":
    sp = L.span_positions(m, b0, span); assert set(hp) <= set(sp) or span == "assistant"
    print(span, "span tokens of the first transcript:", len(sp), repr(m.tok.decode(m.tok(L.chat(m, b0, gen=False), add_special_tokens=False).input_ids[sp[0]:sp[-1] + 1])[:200]), flush=True)
print("empty-mask check: identical; instruction tokens of the first transcript:", len(hp),
      repr(m.tok.decode(m.tok(L.chat(m, b0, gen=False), add_special_tokens=False).input_ids[hp[0]:hp[-1] + 1])), flush=True)
K = np.full(F.shape, np.nan, np.float32); t0 = time.time()
for j, i in enumerate(idx):
    b = L.base_msgs(rows[i])
    hp_i = L.instruction_positions(m, b, instr(rows[i])) if span == "instruction" else L.span_positions(m, b, span)
    K[i] = L.probe_feats(m, b, hide=hp_i)
    if j % 50 == 0:
        print("knockout", j, "of", len(idx), f"{time.time() - t0:.0f}s", flush=True); np.save(os.path.join(out, f"{tag}_feats_knockout{suffix}.npy"), K)
np.save(os.path.join(out, f"{tag}_feats_knockout{suffix}.npy"), K)
print("DONE knockout", len(idx), f"{time.time() - t0:.0f}s", flush=True)
