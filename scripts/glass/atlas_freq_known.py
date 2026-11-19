"""A37 known answer (pre-registered DIARY 13:56): among trusted triggers that use an explicit frequency feature (k >= 2,
either 'cos(k·v)'/'sin(k·v)' or 'k·v mod p'), >= 80% use one of the network's key frequencies (top 5 of the embedding's
Fourier power, from its meta). Frequency 1 (raw values) would count only if 1 were a key frequency.

A40 (pre-registered DIARY 14:18): --basis=log counts a trigger as key only if it uses a key frequency of a discrete-log
feature ('k·log v mod q'); --basis=line only of a number-line feature. Keys for --basis=log: --keys=4,41,28,31,30.

Usage: python3 scripts/glass/atlas_freq_known.py TARGET [ATLAS_JSON] [--basis=log|line] [--keys=...]"""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas


def _freqs(u):
    """(frequency, basis) pairs named by a feature: 'k·log v mod q' -> (k, 'log'); 'k·v mod p' or cos/sin(k·v) -> (k, 'line')."""
    return {(int(f), "log" if "·log " in u else "line") for f in re.findall(r"(\d+)·", u)}


def score(a, key, basis=None):
    n_f, n_key, per = 0, 0, {}
    for k, m in a["matrices"].items():
        c = Counter()
        for x in m["mechanisms"]:
            if x["trigger"]["rule"] in ("always", "never") or atlas._conf(x) < atlas.TRUSTED:
                continue
            fr = {fb for u in x["trigger"].get("features_used", []) for fb in _freqs(u)}
            fr = {(f, b_) for f, b_ in fr if f >= 2 or 1 in key}
            if fr:
                n_f += 1; n_key += any(f in key and (basis is None or b_ == basis) for f, b_ in fr)
                c.update(f"{f}{'L' if b_ == 'log' else ''}" for f, b_ in fr)
        per[k] = (sum(1 for x in m["mechanisms"]), c.most_common(6))
    return n_f, n_key, per


if __name__ == "__main__":
    pos = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--"))
    t = pos[0]
    path = Path(pos[1]) if len(pos) > 1 else ROOT / f"results/glass/atlas/{t}_d4.json"
    a = json.loads(path.read_text())
    meta = json.loads((ROOT / "lab/models" / t / "meta.json").read_text())
    key = set(map(int, opt["keys"].split(","))) if "keys" in opt else set(meta["key_frequencies"][:5])
    basis = opt.get("basis"); nfreq = (113 // 2 - 1) * (2 if basis else 1)
    n_f, n_key, per = score(a, key, basis)
    share = n_key / n_f if n_f else float("nan")
    print(f"{t} ({path.name}): key frequencies {sorted(key)}" + (f" in the {basis} basis (L = log-basis feature)" if basis else ""))
    print(f"Trusted triggers using a k >= 2 frequency feature: {n_f}; using a key frequency: {n_key} "
          f"({100 * share:.0f}%; bar 80%, chance {100 * 5 / nfreq:.0f}%)")
    for k, (n, c) in per.items():
        if c:
            print(f"  {k} ({n} mechanisms): {c}")
    print("KNOWN ANSWER:", "PASS" if n_f and share >= 0.8 else "FAIL")
