"""S5b self-check (counts only): the no-story check must hold on S5b networks and must catch a real story.

  selfcheck_s5b.py <draw>

For every S5b network made from an S1 network: its parent's true blueprint (a real story, on the wrong network) must
not be a rule above the family-wise line there (the network has no such structure), and the same blueprint on the
parent itself must be caught as a story (the check's positive control).
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402


def main():
    draw = sys.argv[1]
    ddir = os.path.join(score.SEALED, draw)
    held, caught, n_held, n_caught = 0, 0, 0, 0
    parents_done = set()
    for net_id in sorted(os.listdir(os.path.join(score.DRAWS, draw))):
        sdir = os.path.join(ddir, net_id)
        meta = os.path.join(sdir, "s5b.json")
        if not net_id.startswith("S5b-") or not os.path.exists(meta):
            continue
        m = json.load(open(meta))
        parent = m.get("of")
        if not parent:
            continue
        pbp = json.load(open(os.path.join(ddir, parent, "true_blueprint.json")))
        with tempfile.TemporaryDirectory() as tmp:
            pbp["task"] = net_id
            f = os.path.join(tmp, "story.json")
            json.dump(pbp, open(f, "w"))
            res = score.s5b_check(os.path.join(score.DRAWS, draw, net_id), f, sdir)
            n_held += 1
            held += all(ok for _n, ok, _s, _l in res)
        if parent not in parents_done:
            parents_done.add(parent)
            pdir = os.path.join(ddir, parent)
            res = score.s5b_check(os.path.join(score.DRAWS, draw, parent), os.path.join(pdir, "true_blueprint.json"), pdir)
            n_caught += 1
            caught += not all(ok for _n, ok, _s, _l in res)
    print(f"{draw} S5b: borrowed stories with no rule above the line on shuffled, untrained and memoriser networks: "
          f"{held} of {n_held}; the same stories caught as rules above the line on their own networks (positive "
          f"control): {caught} of {n_caught}")


if __name__ == "__main__":
    main()
