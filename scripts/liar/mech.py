"""LT1 mechanism step (ledger LT1-mech). For one model: d_L = mean residual(lie) - mean residual(truth) at the
transcript's last token on the train split (hidden_last.npy); add +d_L to honest transcripts and -d_L to lies
at every position of the assistant turn while the transcript is encoded (decoder layer L-1 output), recompute
the 65 probe features with the cached path, and score them with the model's own K1 detector.
Layer: chosen on a train subset (16 lie + 16 truth, seed 2) among every 4th layer, maximising the smaller of the
two gap-closing fractions. Test: 40 lie + 40 truth from the test split (seed 3); controls: 3 random directions
of the same norm (seed 4). Exploratory, not for the claim: the same at the last token only.
Usage: python3 mech.py OUTDIR HF_MODEL TAG DETECTOR_JSON"""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liar_common as L
out, name, tag, detp = sys.argv[1:5]
rows = [json.loads(l) for l in open(os.path.join(out, f"{tag}_transcripts.jsonl"))]
F = np.load(os.path.join(out, f"{tag}_feats.npy")).astype(np.float64)
H = np.load(os.path.join(out, f"{tag}_hidden_last.npy"), mmap_mode="r")
n = min(len(rows), len(F), len(H)); rows, F = rows[:n], F[:n]
det = json.load(open(detp))
mu, sd, w = np.array(det["mean"]), np.array(det["scale"]), np.array(det["coef"]); b0 = det["intercept"]
def score(f): return float(((np.asarray(f) - mu) / sd) @ w + b0)
import score_k
lab = np.array(score_k.relabel(rows)); tr = np.array([r["qi"] in L.TRAIN for r in rows])
NEG = det.get("neg", "truth")              # LT2 M2: failed_lie
lie_tr, tru_tr = np.flatnonzero(tr & (lab == "lie")), np.flatnonzero(tr & (lab == NEG))
lie_te, tru_te = np.flatnonzero(~tr & (lab == "lie")), np.flatnonzero(~tr & (lab == NEG))
nL = H.shape[1] - 1
dvec = {Lh: torch.tensor(np.asarray(H[lie_tr, Lh], np.float32).mean(0) - np.asarray(H[tru_tr, Lh], np.float32).mean(0)).cuda()
        for Lh in range(4, nL - 1, 4)}
m = L.load(name, out)
def run(idx, sign, Lh, vec, last_only=False):
    s = []
    for i in idx:
        msgs = L.base_msgs(rows[i]); pos = L.answer_positions(m, msgs)
        if last_only: pos = pos[-1:]
        s.append(score(L.probe_feats(m, msgs, steer=(Lh - 1, sign * vec, pos))))
    return np.array(s)
_base = {}
def base(idx):
    """Unsteered scores, recomputed in this session with the same code path (bf16 noise cancels)."""
    for i in idx:
        if int(i) not in _base: _base[int(i)] = score(L.probe_feats(m, L.base_msgs(rows[i])))
    return np.array([_base[int(i)] for i in idx])
def closing(idx_t, idx_l, Lh, vec, last_only=False):
    bt, bl = base(idx_t), base(idx_l); gap = bl.mean() - bt.mean()
    st, sl = run(idx_t, +1, Lh, vec, last_only), run(idx_l, -1, Lh, vec, last_only)
    return dict(gap=float(gap), close_plus=float((st.mean() - bt.mean()) / gap), close_minus=float((bl.mean() - sl.mean()) / gap),
                truth_base=float(bt.mean()), truth_steered=float(st.mean()), lie_base=float(bl.mean()), lie_steered=float(sl.mean()))
_m0 = L.base_msgs(rows[0]); _z = torch.zeros(H.shape[2]).cuda()
_u, _s = np.array(L.probe_feats(m, _m0)), np.array(L.probe_feats(m, _m0, steer=(nL // 2, _z, L.answer_positions(m, _m0))))
assert np.abs(_u - _s).max() == 0.0, ("zero steering changed features", float(np.abs(_u - _s).max()))
print("zero-vector hook check: identical", flush=True)
t0 = time.time(); r2 = np.random.default_rng(2)
st_l, st_t = r2.choice(lie_tr, 16, replace=False), r2.choice(tru_tr, 16, replace=False)
sweep = {}
for Lh, v in dvec.items():
    sweep[Lh] = closing(st_t, st_l, Lh, v); print("sweep", Lh, sweep[Lh], f"{time.time() - t0:.0f}s", flush=True)
best = max(sweep, key=lambda k: min(sweep[k]["close_plus"], sweep[k]["close_minus"]))
r3 = np.random.default_rng(3)
te_l, te_t = r3.choice(lie_te, min(40, len(lie_te)), replace=False), r3.choice(tru_te, min(40, len(tru_te)), replace=False)
res = dict(tag=tag, neg=NEG, layer=int(best), d_norm=float(dvec[best].norm()), sweep={str(k): v for k, v in sweep.items()})
res["test"] = closing(te_t, te_l, best, dvec[best]); print("test", res["test"], flush=True)
g4 = torch.Generator(device="cpu").manual_seed(4); rnd = []
for j in range(3):
    r = torch.randn(dvec[best].shape[0], generator=g4).cuda(); r = r / r.norm() * dvec[best].norm()
    rnd.append(closing(te_t, te_l, best, r)); print("random", j, rnd[-1], flush=True)
res["random"] = rnd
res["random_mean_abs_close_plus"] = float(np.mean([abs(x["close_plus"]) for x in rnd]))
res["random_mean_abs_close_minus"] = float(np.mean([abs(x["close_minus"]) for x in rnd]))
res["pass"] = bool(res["test"]["close_plus"] >= 0.5 and res["test"]["close_minus"] >= 0.5 and
                   res["random_mean_abs_close_plus"] < 0.1 and res["random_mean_abs_close_minus"] < 0.1)
res["exploratory_last_token_only"] = closing(te_t, te_l, best, dvec[best], last_only=True)
res["seconds"] = round(time.time() - t0)
print(res, flush=True)
json.dump(res, open(os.path.join(out, f"{tag}_mech_{NEG}.json"), "w"), indent=1)
