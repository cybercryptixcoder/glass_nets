"""D1: score decompiler reports against the hidden answer keys (train.json). The decompiler never reads these; this
script runs after it, only to check it.

  python3 scripts/decomp/score_known.py [ids...]     (default: every results/decomp/<id>.json with a train.json answer key)

For each network:
  law_vs_truth   fraction of all (x, y) for which the recovered program gives the TRUE answer (not the network's)
  structure      for group tasks: is the recovered symbol -> element map a group isomorphism up to translation,
                 i.e. g(x * y) - g(x) - g(y) is the same element for (almost) all pairs (M3)
Writes results/decomp/<id>_score.json.
"""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "decomp"))
from tasks import table

DEC = os.path.join(ROOT, "results", "decomp")


def score(mid):
    rep = json.load(open(os.path.join(DEC, mid + ".json")))
    tj = os.path.join(ROOT, "lab", "models", mid, "train.json")
    if not os.path.exists(tj) or not rep.get("group"):
        return None
    info = json.load(open(tj))
    task = info.get("task") or info.get("op")
    if task is None:
        return None
    if "task" in info:
        T, n, n_out, key = table(task, info.get("p", 113))
    else:                                       # the older modadd/modmul networks
        p = info["p"]; a = np.arange(p)[:, None]; b = np.arange(p)[None, :]
        T = {"add": (a + b) % p, "sub": (a - b) % p, "mul": (a * b) % p}[task]; n = n_out = p; key = {"rule": task}
    if rep.get("cayley_table"):                  # non-abelian (orbit mode): the program is a table over the symbols
        C = np.array(rep["cayley_table"])
        law = float((C == T[:n, :n]).mean())
        # the recovered symbol -> element map is the identity on symbols, so it is a homomorphism exactly where the
        # recovered table agrees with the true one: the isomorphism score equals law_vs_truth here
        out = {"model": mid, "task": task, "rule": key.get("rule"), "hidden": key.get("hidden"), "law_vs_truth": law,
               "structure_isomorphism": law, "group_name": rep.get("group_name"), "element_orders": rep.get("element_orders"),
               "symbols_placed_a": n, "symbols_placed_b": n}
        json.dump(out, open(os.path.join(DEC, mid + "_score.json"), "w"), indent=1)
        return out
    d = np.array(rep["group"])
    ga = {int(k): np.array(v) for k, v in rep.get("element_of_symbol_a", rep.get("element_of_symbol", {})).items()}
    gb = {int(k): np.array(v) for k, v in rep.get("element_of_symbol_b", rep.get("element_of_symbol", {})).items()}
    gc = {int(k): tuple(v) for k, v in rep["element_of_class"].items()}
    inv = {}
    for c, h in gc.items():
        inv.setdefault(h, c)
    left = rep.get("outputs", {}).get("leftover", [])
    absorbing = left[0] if len(left) == 1 else None
    ok = tot = 0
    diffs = {}
    for x in range(n):
        for y in range(n):
            tot += 1
            if (x not in ga or y not in gb) and absorbing is not None:
                ok += absorbing == T[x, y]
            elif x in ga and y in gb:
                s = tuple(((ga[x] + gb[y]) % d).tolist())
                ok += inv.get(s, -1) == T[x, y]
                if T[x, y] in gc:
                    dd = tuple(((np.array(gc[T[x, y]]) - ga[x] - gb[y]) % d).tolist())
                    diffs[dd] = diffs.get(dd, 0) + 1
    out = {"model": mid, "task": task, "rule": key.get("rule"), "hidden": key.get("hidden"),
           "law_vs_truth": ok / tot,
           "structure_isomorphism": max(diffs.values()) / max(1, sum(diffs.values())) if diffs else 0.0,
           "symbols_placed_a": len(ga), "symbols_placed_b": len(gb)}
    json.dump(out, open(os.path.join(DEC, mid + "_score.json"), "w"), indent=1)
    return out


def main():
    ids = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(DEC) if f.endswith(".json") and not f.endswith(("_score.json", "_pow.json")))
    for mid in ids:
        r = score(mid)
        if r:
            print(f"{mid:22s} {r['rule']:40s} law vs truth {r['law_vs_truth']:.3f}   isomorphism {r['structure_isomorphism']:.3f}")


if __name__ == "__main__":
    main()
