"""A37 known answer (pre-registered DIARY 13:56): among trusted triggers that use an explicit frequency feature (k >= 2,
either 'cos(k·v)'/'sin(k·v)' or 'k·v mod p'), >= 80% use one of the network's key frequencies (top 5 of the embedding's
Fourier power, from its meta). Frequency 1 (raw values) would count only if 1 were a key frequency.

Usage: python3 scripts/glass/atlas_freq_known.py TARGET [ATLAS_JSON]   (default results/glass/atlas/TARGET_d4.json)"""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas


def score(a, key):
    n_f, n_key, per = 0, 0, {}
    for k, m in a["matrices"].items():
        c = Counter()
        for x in m["mechanisms"]:
            if x["trigger"]["rule"] in ("always", "never") or atlas._conf(x) < atlas.TRUSTED:
                continue
            fr = {int(f) for u in x["trigger"].get("features_used", []) for f in re.findall(r"(\d+)·", u)}
            fr = {f for f in fr if f >= 2 or 1 in key}
            if fr:
                n_f += 1; n_key += bool(fr & key); c.update(fr)
        per[k] = (sum(1 for x in m["mechanisms"]), c.most_common(6))
    return n_f, n_key, per


if __name__ == "__main__":
    t = sys.argv[1]
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / f"results/glass/atlas/{t}_d4.json"
    a = json.loads(path.read_text())
    meta = json.loads((ROOT / "lab/models" / t / "meta.json").read_text())
    key = set(meta["key_frequencies"][:5]); nfreq = meta["answer"] and (113 // 2 - 1)
    n_f, n_key, per = score(a, key)
    share = n_key / n_f if n_f else float("nan")
    print(f"{t} ({path.name}): key frequencies {sorted(key)}")
    print(f"Trusted triggers using a k >= 2 frequency feature: {n_f}; using a key frequency: {n_key} "
          f"({100 * share:.0f}%; bar 80%, chance {100 * 5 / nfreq:.0f}%)")
    for k, (n, c) in per.items():
        if c:
            print(f"  {k} ({n} mechanisms): {c}")
    print("KNOWN ANSWER:", "PASS" if n_f and share >= 0.8 else "FAIL")
