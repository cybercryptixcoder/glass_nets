"""Ledger LT5: does every probe group read the instruction? Analysis only, on saved features.

Usage: python3 -I score_groups.py DATA_DIR OUT_JSON TAG_A TAG_B [TAG ...]
  TAG_A and TAG_B are the two clean families (scored within and across); any further tags are scored within only.
  Set LIAR_FULL=1 to use full-path test features (score_k.load), as ledger LT1-full requires for claims.

Groups are the probe_type labels of probes.csv (the paper's own); a group needs at least 3 probes.
For each group, the K1 detector of score_k.py (standardise, logistic regression C = 1, lie vs truth on the train
split) is fit on that group's columns only and scored on the test split: lie vs truth, failed_lie vs truth
(K2 i) and failed_lie vs truth on the instruction-deleted features (control).
Reference: 20 random probe sets of the same size (seed 5), median failed_lie vs truth within family.
Known case (checked at start): the group of all 65 probes reproduces score_k's within-family AUCs exactly.
Reading (fixed in LT5 before any score): a group reads the instruction if failed_lie vs truth >= 0.8 within both
clean families; headline: New_unknowable >= 0.8 in both with its deleted-instruction control <= 0.65.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_k as S


def sub(D, cols):
    E = dict(D)
    E["F"] = D["F"][:, cols]
    E["Fn"] = D["Fn"][:, cols]
    return E


def scores(Dtr, Dte, cols):
    M = S.fit(sub(Dtr, cols))
    E = sub(Dte, cols)
    lt, _ = S.auc(M, E, "lie", "truth")
    ft, n = S.auc(M, E, "failed_lie", "truth")
    fn, _ = S.auc(M, E, "failed_lie", "truth", feats="Fn")
    return dict(lie_vs_truth=lt, failed_lie_vs_truth=ft, failed_vs_truth_instruction_deleted=fn, n_ft=n)


def main():
    d, out, ta, tb = sys.argv[1:5]
    rest = sys.argv[5:]
    probes = pd.read_csv(os.path.join(d, "probes.csv"))
    types = probes["probe_type"].astype(str).tolist()
    groups = {t: [j for j, x in enumerate(types) if x == t] for t in sorted(set(types))}
    groups = {t: c for t, c in groups.items() if len(c) >= 3}
    allc = list(range(len(types)))
    D = {t: S.load(d, t) for t in [ta, tb] + rest}

    # known case: all 65 probes reproduce score_k's within-family numbers exactly
    for t in [ta, tb]:
        M = S.fit(D[t])
        ref = (S.auc(M, D[t], "lie", "truth")[0], S.auc(M, D[t], "failed_lie", "truth")[0])
        got = scores(D[t], D[t], allc)
        assert (got["lie_vs_truth"], got["failed_lie_vs_truth"]) == ref, (t, got, ref)
    print("known case: all-probe group reproduces score_k exactly", flush=True)

    res = {"full_path": os.environ.get("LIAR_FULL") == "1", "groups": {t: [probes["probe"][j] for j in c] for t, c in groups.items()},
           "within": {}, "across": {}, "random_reference": {}}
    rng = np.random.default_rng(5)
    for g, cols in list(groups.items()) + [("all", allc)]:
        res["within"][g] = {t: scores(D[t], D[t], cols) for t in D}
        res["across"][g] = {f"{ta} -> {tb}": scores(D[ta], D[tb], cols), f"{tb} -> {ta}": scores(D[tb], D[ta], cols)}
        if g != "all":
            draws = [sorted(rng.choice(len(types), size=len(cols), replace=False).tolist()) for _ in range(20)]
            res["random_reference"][g] = {t: float(np.median([scores(D[t], D[t], c)["failed_lie_vs_truth"] for c in draws]))
                                          for t in [ta, tb]}
    ok = lambda v, thr: v is not None and v >= thr
    res["reads_instruction"] = {g: all(ok(res["within"][g][t]["failed_lie_vs_truth"], 0.8) for t in [ta, tb]) for g in res["within"]}
    nu = res["within"].get("New_unknowable")
    if nu:
        res["headline_new_unknowable"] = bool(all(ok(nu[t]["failed_lie_vs_truth"], 0.8) for t in [ta, tb]) and
                                              all(nu[t]["failed_vs_truth_instruction_deleted"] is not None and
                                                  nu[t]["failed_vs_truth_instruction_deleted"] <= 0.65 for t in [ta, tb]))
    json.dump(res, open(out, "w"), indent=1)
    for g in res["within"]:
        w = res["within"][g]
        print(f"{g:16s} n={len(groups.get(g, allc)):2d} " + " | ".join(
            f"{t}: ft {w[t]['failed_lie_vs_truth']:.3f} del {w[t]['failed_vs_truth_instruction_deleted']:.3f} lt {w[t]['lie_vs_truth']:.3f}" for t in D)
              + "  across ft " + " / ".join(f"{v['failed_lie_vs_truth']:.3f}" for v in res["across"][g].values())
              + ("  rand " + " / ".join(f"{v:.3f}" for v in res["random_reference"][g].values()) if g in res["random_reference"] else ""))
    print("reads_instruction", res["reads_instruction"])
    print("headline_new_unknowable", res.get("headline_new_unknowable"))


def main_t():
    """Ledger LT7 (GROUP_T=1): the T1 detector (lie vs failed_lie, score_t.py's procedure) fit per probe group,
    within each family. Reading: carries the real-lie trace if T1 >= 0.75 in both families; none if <= 0.65 in both."""
    d, out, ta, tb = sys.argv[1:5]
    rest = sys.argv[5:]
    probes = pd.read_csv(os.path.join(d, "probes.csv"))
    types = probes["probe_type"].astype(str).tolist()
    groups = {t: [j for j, x in enumerate(types) if x == t] for t in sorted(set(types))}
    groups = {t: c for t, c in groups.items() if len(c) >= 3}
    groups["all"] = list(range(len(types)))
    D = {t: S.load(d, t) for t in [ta, tb] + rest}

    def t1(t, cols):
        M = S.fit(sub(D[t], cols), neg="failed_lie")
        a, n = S.auc(M, sub(D[t], cols), "lie", "failed_lie")
        return dict(T1_lie_vs_failed=a, n=n)

    # known case: all 65 probes reproduce score_t's within-family T1 exactly (same fit and AUC calls)
    for t in [ta, tb]:
        M = S.fit(D[t], neg="failed_lie")
        assert t1(t, groups["all"])["T1_lie_vs_failed"] == S.auc(M, D[t], "lie", "failed_lie")[0]
    print("known case: all-probe group reproduces score_t's T1 exactly", flush=True)
    res = {"full_path": os.environ.get("LIAR_FULL") == "1", "T1_within": {g: {t: t1(t, c) for t in D} for g, c in groups.items()}}
    v = lambda g, t: res["T1_within"][g][t]["T1_lie_vs_failed"]
    res["carries_real_lie"] = {g: all(v(g, t) >= 0.75 for t in [ta, tb]) for g in groups}
    res["carries_none"] = {g: all(v(g, t) <= 0.65 for t in [ta, tb]) for g in groups}
    json.dump(res, open(out, "w"), indent=1)
    for g in groups:
        print(f"{g:16s} n={len(groups[g]):2d} " + " | ".join(f"{t}: T1 {v(g, t):.3f}" for t in D))
    print("carries_real_lie", res["carries_real_lie"])
    print("carries_none", res["carries_none"])


if __name__ == "__main__":
    main_t() if os.environ.get("GROUP_T") == "1" else main()
