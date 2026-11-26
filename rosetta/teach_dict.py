"""Teachability, live: a fresh DeepSeek Flash call gets ONLY a plain-English definition of one of LD2's translated
variables and 8 held-out positions where the variable is active, and must name the 3 squares where it is most active.
Scored against the variable's real top-3 squares; a rival sentence (another variable's definition) is the control.
Usage: python3 rosetta/teach_dict.py"""
import glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import llm  # noqa: E402
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L  # noqa: E402

SENT = {  # plain-English renderings of the programs (written from the program text alone)
    111: "a square on the fourth rank or beyond (counting from our side) that no enemy pawn can ever attack, and that at least four of our legal moves can reach",
    87: "the square of our own king when it is attacked (we are in check)",
    174: "the square our best move goes to, when an enemy piece (not a pawn) stands there and it has more attackers than defenders",
    19: "a square next to our king, holding an undefended enemy piece that we attack, where the piece standing there attacks five or more squares",
    14: "a square attacked by one of our pawns, holding a piece that is under attack by its opponent, that at least one of our legal moves can reach",
}

if __name__ == "__main__":
    E = {}
    for f in glob.glob(f"{L.RES}/dict_s{L.SITE}_w*.json"):
        for e in json.load(open(f)):
            E[e["feature"]] = e
    rng = np.random.default_rng(3)
    jobs = []
    feats = list(SENT)
    for j, f in enumerate(feats):
        rows = [r for r in L.TE if F.fmap(f, [r]).max() > 0]
        rows = list(rng.choice(rows, 8, replace=False))
        act = F.fmap(f, rows)
        fens = [L.items[r]["fen"] for r in rows]
        top = [set(llm.real_name(s, fn) for s in np.argsort(-a)[:3]) for a, fn in zip(act, fens)]
        rival = SENT[feats[(j + 1) % len(feats)]]
        jobs.append((f, SENT[f], rival, fens, top))

    def run(job):
        f, s, rv, fens, top = job
        def score(sentence):
            pred = llm.teach(sentence, fens, k=3, effort="low")
            if len(pred) != len(top):
                return None
            return float(np.mean([len(set(p[:3]) & t) / 3 for p, t in zip(pred, top)]))
        return {"feature": f, "definition_program": E[f]["text"], "fidelity": E[f]["heldout_fidelity"], "sentence": s,
                "hit_rate": score(s), "rival_hit_rate": score(rv), "chance": 3 / 64}
    with ThreadPoolExecutor(5) as ex:
        out = list(ex.map(run, jobs))
    for o in out:
        print(json.dumps(o))
    json.dump(out, open(f"{L.RES}/teach_dict_s{L.SITE}.json", "w"), indent=1)
