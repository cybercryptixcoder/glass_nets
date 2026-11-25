"""Score the frozen KR1b (B) rule (results/kronecker/FROZEN_RULE.json) on held-out n against P4's missed zeros.
Usage: rule_heldout.py n...   writes results/kronecker/rule_heldout.json"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_search as rs                     # applies the P4 patch to face_reduce on import
fro = json.load(open("results/kronecker/FROZEN_RULE.json"))
F = fro["features"]
def holds(atom, x):
    if "==" in atom: k, c = atom.split("=="); return x[F.index(k)] == int(c)
    k, c = atom.split(">="); return x[F.index(k)] >= int(c)
def rule(x): return any(all(holds(a, x) for a in conj) for conj in fro["rule_or_of_conjunctions"])
out = json.load(open("results/kronecker/rule_heldout.json")) if os.path.exists("results/kronecker/rule_heldout.json") else {}
for n in map(int, sys.argv[1:]):
    X, W, Y, N = rs.build([n])
    flag = np.array([rule(x) for x in X], dtype=bool)
    parts, g = rs.load(n); nonzero = int((g != 0).sum()); del g
    resid = int(W[Y == 1].sum()); caught = int(W[flag & (Y == 1)].sum()); fz = int(W[flag & (Y == 0)].sum())
    r = dict(n=n, p4_missed_zeros=resid, caught=caught, caught_frac=caught / max(1, resid), false_zero_triples=fz,
             false_zero_frac_of_nonzero=fz / nonzero, meets=bool(caught >= 0.5 * resid and fz <= 0.001 * nonzero))
    out[str(n)] = r; print(r, flush=True)
    json.dump(out, open("results/kronecker/rule_heldout.json", "w"), indent=1)
