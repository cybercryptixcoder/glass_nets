"""A44 known answer (pre-registered DIARY 22:14): among rule-labelled W_out / W_U mechanisms whose trigger reads a frequency,
more than half write the frequency they read (same frequency and basis); at least 15 such mechanisms. Annotates the saved
atlas in place (A44 fields only; no number changes), using the trees in its model file and the vocabulary it was built with.
Usage: python3 scripts/glass/atlas_writes_known.py TARGET VOCAB [--keys=..]   (VOCAB: angle | both)"""
import json, os, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
pos = [x for x in sys.argv[1:] if not x.startswith("--")]; opt = dict((x[2:].split("=", 1) + [""])[:2] for x in sys.argv[1:] if x.startswith("--"))
t, vocab = pos[0], pos[1]
os.environ["ATLAS_MODADD_VOCAB"] = vocab
sys.path.insert(0, str(ROOT / "lab")); import atlas
path = ROOT / f"results/glass/atlas/{opt.get('file', t + '_d4')}.json"
a = json.loads(path.read_text()); P = pickle.loads(path.with_name(path.stem + "_model.pkl").read_bytes())
tgt = atlas.make_target(t)
atlas.annotate_writes(a, tgt, P["trees"], lambda k: tgt._feats_uncached(tgt.X[:4], k)[0])
path.write_text(json.dumps(a, indent=1))
meta = json.loads((ROOT / "lab/models" / t / "meta.json").read_text())
keys = set(map(int, opt["keys"].split(","))) if opt.get("keys") else set(meta["key_frequencies"][:5])
n = match = keyw = 0
for k in ("W_out", "W_U"):
    for x in a["matrices"][k]["mechanisms"]:
        r = x["trigger"].get("reads_frequency")
        if x["trigger"].get("label") != "rule" or not r:
            continue
        wv = x["answer_wave"]; n += 1
        match += (wv["frequency"], wv["basis"]) == (r["frequency"], r["basis"]); keyw += wv["frequency"] in keys
print(f"{t}: rule-labelled W_out/W_U mechanisms with a frequency trigger: {n}; write the frequency they read: {match} "
      f"({100 * match / max(n, 1):.0f}%; bar > 50%, at least 15); write a key frequency: {keyw} ({100 * keyw / max(n, 1):.0f}%)")
print("KNOWN ANSWER:", "PASS" if n >= 15 and match / n > 0.5 else "FAIL")
