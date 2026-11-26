"""Rosetta on the known-answer networks. The engine sees only each network's activity and the chess library; the
sealed rules are read afterwards, by score_yardstick(), to grade what the engine reported.
Usage: python3 rosetta/run_yardstick.py [stage]   stage: controls | nets | score"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta.engine import Defs, spec_map, acceptance_line, restore_share  # noqa: E402
from rosetta.yardstick import Small  # noqa: E402
K = os.path.expanduser("~/kdata"); Y = os.path.join(K, "rosetta", "yard"); RES = "results/rosetta"
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(2)
names = json.load(open(os.path.join(K, "rosetta", "prim_names.json")))
P = np.load(os.path.join(K, "rosetta", "prims_LD2.npy"), mmap_mode="r")
sp = np.load(os.path.join(Y, "split.npz")); tr, te = sp["tr"], sp["te"]
X = np.load(os.path.join(Y, "planes.npy"), mmap_mode="r")


def load(name, seed=None):
    m = Small()
    if seed is None:
        m.load_state_dict(torch.load(os.path.join(Y, f"{name}.pt")))
    else:
        torch.manual_seed(seed)
    return m.eval()


@torch.no_grad()
def run(m, rows, edit=None):
    x = torch.tensor(np.asarray(X[rows], dtype=np.float32))
    y, h = m(x, edit=edit, return_last=True)
    return y.numpy(), h.flatten(2).numpy()


def units_for(m, D, n_keep=6):
    rows = D.rt[:1500]
    y0, h0 = run(m, rows)
    mean = h0.mean(0)
    eff = []
    for c in range(h0.shape[1]):
        def ed(h, c=c):
            h = h.clone(); h[:, c] = torch.tensor(mean[c]).view(8, 8); return h
        y1, _ = run(m, rows, ed)
        eff.append(float(np.mean(np.abs(y1 - y0))))
    order = [int(c) for c in np.argsort(eff)[::-1][:n_keep]]
    # the readout direction: per-square projection onto the channel weights that best predict the output
    ytr, htr = run(m, D.rs)
    T = htr.sum(2)
    w = np.linalg.solve(T.T @ T + 1e-3 * np.eye(T.shape[1]), T.T @ (ytr - ytr.mean()))
    return order, eff, w


def define_net(m, D, tag):
    order, eff, w = units_for(m, D)
    out = {"net": tag, "ablation_effect": [round(e, 4) for e in eff], "units": []}
    for c in order:
        d = D.define(lambda rows, c=c: run(m, rows)[1][:, c])
        d.update({"unit": f"channel {c}", "channel": c, "effect": round(eff[c], 4)})
        out["units"].append(d)
    d = D.define(lambda rows: np.einsum("c,ncs->ns", w, run(m, rows)[1]))
    d.update({"unit": "readout direction", "channel": None, "effect": None})
    out["units"].append(d)
    return out, w


def substitution(m, D, unit):
    if unit["channel"] is None or unit["program"] is None:
        return None
    c = unit["channel"]; rows = D.rt[:1500]
    y0, h0 = run(m, rows); mean = h0.mean(0)
    pm = spec_map(np.asarray(P[rows], dtype=np.float32), unit["program"])
    a = np.polyfit(pm.reshape(-1), h0[:, c].reshape(-1), 1)
    def abl(h):
        h = h.clone(); h[:, c] = torch.tensor(mean[c]).view(8, 8); return h
    def sub(h):
        h = h.clone(); h[:, c] = torch.tensor((a[1] + a[0] * pm).reshape(-1, 8, 8), dtype=h.dtype); return h
    return round(restore_share(y0, run(m, rows, abl)[0], run(m, rows, sub)[0]), 4)


if __name__ == "__main__":
    stage = sys.argv[1]
    D = Defs(P, names, tr, te)
    t0 = time.time()
    if stage == "controls":                       # the line: an untrained network (seed 11) + square-shuffled units
        m = load("UNTRAINED", seed=11)
        ctrl = []
        for c in range(32):
            d = D.define(lambda rows, c=c: run(m, rows)[1][:, c]); d["unit"] = f"untrained(seed 11) channel {c}"; ctrl.append(d)
            print(c, round(time.time() - t0), d["text"], d["heldout_fidelity"], flush=True)
        for name in ["S1", "S2", "S3"]:
            mm = load(name); rng = np.random.default_rng(5)
            for c in range(4):
                def shuf(rows, c=c, mm=mm):
                    a = run(mm, rows)[1][:, c]; return np.stack([r[rng.permutation(64)] for r in a])
                d = D.define(shuf); d["unit"] = f"{name} channel {c}, squares shuffled"; ctrl.append(d)
        mx, q99 = acceptance_line(ctrl); px, pq = acceptance_line(ctrl, "pos")
        json.dump({"line_max": mx, "line_q99": q99, "line_pos_max": px, "line_pos_q99": pq, "controls": ctrl},
                  open(f"{RES}/yard_controls.json", "w"), indent=1)
        print("LINE per-square max", round(mx, 4), "| position-level max", round(px, 4), round(time.time() - t0), "s")
    elif stage == "nets":
        cj = json.load(open(f"{RES}/yard_controls.json")); line, pline = cj["line_max"], cj["line_pos_max"]
        allout = {"line": line, "line_pos": pline}
        for name, seed in [("S1", None), ("S2", None), ("S3", None), ("MEM", None), ("UNTRAINED2", 12)]:
            m = load(name if seed is None else "UNTRAINED", seed=seed)
            out, _ = define_net(m, D, name)
            cands = []
            for u in out["units"]:
                u["accepted_sq"] = bool(u["program"] is not None and abs(u["heldout_fidelity"]) >= line)
                u["accepted_pos"] = bool(u["pos"]["program"] is not None and abs(u["pos"]["heldout_fidelity"]) >= pline)
                u["accepted"] = u["accepted_sq"] or u["accepted_pos"]
                u["restore"] = substitution(m, D, u) if u["accepted_sq"] else None
                if u["accepted_pos"]:
                    cands.append((u["unit"] == "readout direction", 1, abs(u["pos"]["heldout_fidelity"]), {**u["pos"], "unit": u["unit"], "level": "position"}))
                if u["accepted_sq"]:
                    cands.append((u["unit"] == "readout direction", 0, abs(u["heldout_fidelity"]), {"program": u["program"], "text": u["text"], "unit": u["unit"], "level": "square", "heldout_fidelity": u["heldout_fidelity"]}))
            acc = [u for u in out["units"] if u["accepted"]]
            out["answer"] = max(cands, key=lambda c: c[:3])[3] if cands else None
            allout[name] = out
            print(name, "accepted", len(acc), "answer:", out["answer"]["text"] if out["answer"] else None, round(time.time() - t0), "s", flush=True)
        json.dump(allout, open(f"{RES}/yard_engine.json", "w"), indent=1)
    elif stage == "score":                        # only now is the sealed file read
        from rosetta.sealed.yardstick_rules import RULES
        eng = json.load(open(f"{RES}/yard_engine.json")); score = {"line": eng["line"]}
        Pt = np.asarray(P[D.rt], dtype=np.float32)
        for name in ["S1", "S2", "S3", "MEM", "UNTRAINED2"]:
            a = eng[name]["answer"]
            if name in RULES:
                truth = np.load(os.path.join(Y, f"{name}_truthmap.npy"))[D.rt].astype(np.float32).reshape(-1)
                r = float(np.corrcoef(spec_map(Pt, a["program"]).reshape(-1), truth)[0, 1]) if a else 0.0
                score[name] = {"sealed_rule": RULES[name], "rosetta_answer": a["text"] if a else None,
                               "corr_with_sealed_rule": round(r, 4), "recovered (>= 0.9)": r >= 0.9}
            else:
                score[name] = {"must_fail": True, "accepted_definitions": sum(u["accepted"] for u in eng[name]["units"]),
                               "passed (none accepted)": eng[name]["answer"] is None}
            print(name, score[name])
        json.dump(score, open(f"{RES}/yard_score.json", "w"), indent=1)
