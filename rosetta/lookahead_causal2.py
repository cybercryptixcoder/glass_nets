"""Causal test of look-ahead (in the spirit of Jenner et al. 2024): at a site, mean-ablate the network's state on ONE
square and measure how much the probability of the network's own chosen move drops. Squares compared, per position:
  t3   = the destination of the network's own follow-up move (ply 3), when it is not a square of the first two moves;
  ctrl = a random square of the same kind (empty / own piece / enemy piece) that is not on the line;
  t1   = the destination of the chosen move itself (reference).
If the network looks ahead, ablating t3 must hurt the current move more than ablating a matched control square.
Usage: python3 rosetta/lookahead_causal.py SITE [n_positions]"""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from kepler.targets.chess.leela import move_index  # noqa: E402
import torch.nn.functional as Fn

if __name__ == "__main__":
    site = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    t0 = time.time()
    NET = os.environ.get("NET", "LD2")
    lines = json.load(open(os.path.join(L.R, f"{NET}_lines.json")))
    rng = np.random.default_rng(site)
    pool = rng.permutation(L.TE)
    names = L.names; Pall = L.P if NET == "LD2" else np.load(os.path.join(L.R, f"prims_{NET}.npy"), mmap_mode="r")
    ix = {k: names.index(k) for k in ("la_m1_from", "la_m1_to", "la_m2_from", "la_m2_to", "la_m3_from", "la_m3_to", "empty",
                                      "la_m1_attacks", "att_us", "att_them")}
    own = [names.index(f"own_{p}") for p in "PNBRQK"]; opp = [names.index(f"opp_{p}") for p in "PNBRQK"]
    rows, t3s, ctrls, t1s, moves, att3, kinds = [], [], [], [], [], [], []
    for i in pool:
        if len(rows) >= n:
            break
        ln = lines[i]
        if len(ln) < 3 or any(m is None for m in ln[:3]):
            continue
        p = np.asarray(Pall[i], dtype=np.float32)
        sq = lambda k: int(np.argmax(p[:, ix[k]])) if p[:, ix[k]].max() > 0 else None
        t3 = sq("la_m3_to"); used = {sq("la_m1_from"), sq("la_m1_to"), sq("la_m2_from"), sq("la_m2_to"), sq("la_m3_from")}
        if t3 is None or t3 in used:
            continue
        occ = lambda s: 0 if p[s, ix["empty"]] > 0 else (1 if p[s, own].max() > 0 else 2)
        tight = lambda s: (occ(s), int(p[s, ix["la_m1_attacks"]] > 0), min(int(p[s, ix["att_us"]]), 2), min(int(p[s, ix["att_them"]]), 2))
        c1 = [s for s in range(64) if s not in used and s != t3 and occ(s) == occ(t3)]
        c2 = [s for s in c1 if tight(s) == tight(t3)]
        if not c1 or not c2:
            continue
        rows.append(i); t3s.append(t3); ctrls.append((int(rng.choice(c1)), int(rng.choice(c2)))); t1s.append(sq("la_m1_to")); moves.append(ln[0])
        att3.append(int(p[t3, ix["la_m1_attacks"]] > 0))
        b3 = chess.Board(L.items[i]["fen"]); b3.push(chess.Move.from_uci(ln[0])); b3.push(chess.Move.from_uci(ln[1])); m3 = chess.Move.from_uci(ln[2])
        kinds.append("check" if b3.gives_check(m3) else "capture" if b3.is_capture(m3) else "quiet")
    rows = np.array(rows)
    m = L.get_net(NET)
    if NET == "LD2":
        A = np.load(os.path.join(L.C, "LD2", f"acts_site{site}.npy"), mmap_mode="r")
        mean = np.asarray(A[L.TR[:3000]], dtype=np.float32).reshape(-1, 128, 64).mean(0)       # [128, 64]
    else:
        mean = L.site_acts(m, L.TR[:2000], site=site).reshape(-1, 128, 64).mean(0)
    boards = [chess.Board(L.items[i]["fen"]) for i in rows]
    midx = [move_index(m, b, chess.Move.from_uci(u)) for b, u in zip(boards, moves)]

    @torch.no_grad()
    def logp(squares):
        out = []
        for j in range(0, len(rows), 500):
            sl = slice(j, j + 500); sqs = squares[sl] if squares is not None else None
            def ed(s, h, sqs=sqs):
                if s == site and sqs is not None:
                    h = h.clone()
                    for b, q in enumerate(sqs):
                        h[b, :, q // 8, q % 8] = torch.tensor(mean[:, q])
                return h
            lg, _ = m(L.planes(rows[sl]), edit=ed)
            for b, (bd, mi) in enumerate(zip(boards[sl], midx[sl])):
                legal = [move_index(m, bd, mm) for mm in bd.legal_moves]
                lp = Fn.log_softmax(lg[b, legal], 0)
                out.append(float(lp[legal.index(mi)]))
        return np.array(out)
    base = logp(None)
    d = {k: base - logp(np.array(v)) for k, v in (("t3", t3s), ("ctrl_occupancy", [c[0] for c in ctrls]), ("ctrl_tight", [c[1] for c in ctrls]), ("t1", t1s))}
    att3 = np.array(att3)
    split = {}
    kk = np.array(kinds)
    for nm, msk in (("t3 attacked by the moved piece", att3 == 1), ("t3 not attacked by the moved piece", att3 == 0),
                    ("follow-up is a check", kk == "check"), ("follow-up is a capture", kk == "capture"), ("follow-up is quiet", kk == "quiet")):
        if msk.sum() > 30:
            dd = d["t3"][msk] - d["ctrl_tight"][msk]
            bb = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            split[nm] = {"n": int(msk.sum()), "t3_minus_tight": round(float(dd.mean()), 4), "ci95": [round(float(np.quantile(bb, .025)), 4), round(float(np.quantile(bb, .975)), 4)]}
    diff = d["t3"] - d["ctrl_tight"]
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    out = {"net": NET, "site": site, "n": int(len(rows)), "mean_drop_logp": {k: round(float(v.mean()), 4) for k, v in d.items()},
           "t3_minus_ctrl": round(float(diff.mean()), 4), "ci95": [round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)],
           "share_t3_hurts_more": round(float((diff > 0).mean()), 4), "split": split,
           "t3_minus_occupancy_ctrl": round(float((d["t3"] - d["ctrl_occupancy"]).mean()), 4)}
    print(json.dumps(out), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/lookahead_causal2_{NET}_site{site}_bytype.json", "w"), indent=1)
