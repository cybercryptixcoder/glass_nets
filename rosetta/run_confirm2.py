"""Replication in a second network (744706) and the live teachability test for accepted LD2 concepts.
Usage: python3 rosetta/run_confirm2.py rep | teach"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs, spec_map  # noqa: E402
from rosetta.core import fidelity, residualizer  # noqa: E402
RES = L.RES; SITE = L.SITE


def concepts():
    dd = json.load(open(f"{RES}/ld2_define_site{SITE}.json")); out = []
    for u in dd["units"]:
        if u.get("accepted_sq"):
            out.append({"unit": u["unit"], "level": "square", "program": u["program"], "text": u["text"], "fid": u["heldout_fidelity"], "channel": u["channel"]})
        if u.get("accepted_pos"):
            out.append({"unit": u["unit"], "level": "position", "program": u["pos"]["program"], "text": u["pos"]["text"], "fid": u["pos"]["heldout_fidelity"], "channel": u["channel"]})
    pj = f"{RES}/ld2_propose_site{SITE}.json"
    if os.path.exists(pj):
        for b in json.load(open(pj))["best"]:
            if b["accepted"]:
                out.append({"unit": b["unit"], "level": "square" if b["level"] == "sq" else "position", "expression": b["expression"], "text": b["expression"], "fid": b["heldout_fidelity"], "channel": b["channel"]})
    return out, dd


def cmap(Pt, c):
    from rosetta.core import expr_map
    if "expression" in c:
        return expr_map(Pt, L.names, c["expression"]).reshape(Pt.shape[0], 64)
    return spec_map(Pt, c["program"])


def rep():
    cs, dd = concepts()
    D = Defs(L.P, L.names, L.TR, L.TE)
    P2 = np.load(os.path.join(L.R, "prims_744706.npy"), mmap_mode="r")
    m2 = L.get_net("744706")
    rows = np.concatenate([D.rs, D.rt]); H = L.site_acts(m2, rows); Hs, Ht = H[:len(D.rs)], H[len(D.rs):]
    Pt2 = np.asarray(P2[D.rt], dtype=np.float32)
    pc = [L.names.index(n) for n in L.names if n[:4] in ("own_", "opp_") and len(n) == 5]
    res_t = residualizer(Pt2[:, :, pc].sum(1))
    out = []
    for c in cs:
        mp = cmap(Pt2, c); best = (0, None)
        for ch in range(128):
            mu = Hs[:, ch].mean(0, keepdims=True); yt = Ht[:, ch] - mu
            if c["level"] == "square":
                f = fidelity(yt.reshape(-1), mp.reshape(-1))
            else:
                f = fidelity(res_t(yt.sum(1).astype(np.float64)), res_t(mp.sum(1).astype(np.float64)))
            if abs(f) > abs(best[0]):
                best = (f, ch)
        line = dd["line"] if c["level"] == "square" else dd["line_pos"]
        r = {**c, "rep_744706_channel": best[1], "rep_744706_fidelity": round(float(best[0]), 4), "replicates": abs(best[0]) >= line}
        out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f"{RES}/ld2_replication_site{SITE}.json", "w"), indent=1)


def teach(n_pos=10, k=5):
    from rosetta import llm
    cs, dd = concepts()
    D = Defs(L.P, L.names, L.TR, L.TE)
    rng = np.random.default_rng(1); rows = rng.choice(D.rt, n_pos, replace=False)
    fens = [L.items[i]["fen"] for i in rows]
    out = []
    sq_concepts = [c for c in cs if c["level"] == "square" and c["channel"] is not None]
    for j, c in enumerate(sq_concepts):
        sent = "; ".join(llm.MEANING.get(t.split(">=")[0].split("<=")[0].replace("not ", ""), t) + (" (" + t + ")") for t in c["text"].split(" and "))
        rival = sq_concepts[(j + 1) % len(sq_concepts)] if len(sq_concepts) > 1 else None
        act = np.asarray(L.A[rows][:, c["channel"] * 64:(c["channel"] + 1) * 64], dtype=np.float32)
        act = act - np.asarray(L.A[D.rs][:, c["channel"] * 64:(c["channel"] + 1) * 64], dtype=np.float32).mean(0)
        top = [set(llm.real_name(s, f) for s in (np.argsort(-a)[:k] if c["fid"] > 0 else np.argsort(a)[:k])) for a, f in zip(act, fens)]
        def score(sentence):
            pred = llm.teach(sentence, fens, k=k)
            hits = [len(set(p[:k]) & t) / k for p, t in zip(pred, top)] if len(pred) == len(top) else []
            return float(np.mean(hits)) if hits else None
        r = {"unit": c["unit"], "definition": c["text"], "sentence": sent, "hit_rate": score(sent),
             "rival_sentence_hit_rate": score("; ".join(rival["text"].split(" and "))) if rival else None, "chance": k / 64}
        out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f"{RES}/ld2_teach_site{SITE}.json", "w"), indent=1)


if __name__ == "__main__":
    rep() if sys.argv[1] == "rep" else teach()
