"""Gather every Kepler result into results/kepler/summary.json (read by the lab page /kepler and by docs).

    python3 scripts/kepler/report.py
"""
import json
import os

import numpy as np

R = "results/kepler"


def load(p, default=None):
    path = os.path.join(R, p)
    return json.load(open(path)) if os.path.exists(path) else default


def chess_boards(items_path, idxs):
    import chess
    meta = json.load(open(items_path))
    items = meta["items"]
    out = []
    for i in idxs:
        it = items[i]
        b = chess.Board(it["fen"])
        mv = chess.Move.from_uci(it["top_move"])
        out.append({"fen": it["fen"], "q": it["q"], "move": b.san(mv), "p": it["top_p"], "source": it["source"]})
    return out


def main():
    s = {}
    # ---------------- fluid (vorticity video model)
    kt, kv, reg, mech, blobs = (load("fluid/k1_known_test.json"), load("fluid/k1_verdict.json"),
                                load("fluid/law_by_regime.json", {}), load("fluid/mechanism.json"),
                                load("fluid/biot_savart_blobs.json"))
    if kv:
        s["fluid"] = {
            "verdict": kv["verdict"], "one_step_err": kv["one_step_rel_err_of_change"],
            "law": kv["law"]["coefs"], "law_r2": kv["law"]["r2_val"], "true": kv["true_coefs"],
            "r2_true_law": kv["r2_of_true_law_on_model"],
            "law_ood": kv["law_ood"]["coefs"], "law_ood_r2": kv["law_ood"]["r2_val"],
            "known": {k: kt[k].get("pass") for k in ("a", "b", "c")} if kt else None,
            "regimes": {k: v for k, v in reg.items() if not k.startswith(("amplitude", "scale"))},
            "amplitude": {k.replace("amplitude_x", ""): v for k, v in reg.items() if k.startswith("amplitude")},
            "scale": {k.replace("scale_kpeak_", ""): v for k, v in reg.items() if k.startswith("scale")},
        }
        if mech:
            at = mech["atlas"]
            s["fluid"]["atlas"] = {q: at[q] for q in at}
            s["fluid"]["swap"] = mech["velocity_swap"]
            s["fluid"]["velocity_site"] = mech["velocity_site"]
        if blobs:
            s["fluid"]["biot_savart"] = blobs
    # ---------------- smoke (dye) video model
    dv, dk = load("dye/k2_verdict.json"), load("dye/k2_known_test.json")
    if dv:
        s["smoke"] = {"verdict": dv["verdict"], "criteria": dv["criteria"], "best_site": dv["best_site"],
                      "probe_trained": [(e["r2_u"] + e["r2_v"]) / 2 for e in dv["probe_trained"]],
                      "probe_untrained": [(e["r2_u"] + e["r2_v"]) / 2 for e in dv["probe_untrained"]],
                      "law_model_velocity": dv["law_model_velocity"]["coefs"], "law_model_velocity_r2": dv["law_model_velocity"]["r2_val"],
                      "law_true_velocity": dv["law_true_velocity"]["coefs"], "law_true_velocity_r2": dv["law_true_velocity"]["r2_val"],
                      "true": dv["true"], "known": {"a": dk["pass_a"], "b": dk["pass_b"]} if dk else None}
    # ---------------- chess
    atlas, k3, lc, calc = (load("chess/atlas_LD2.json"), load("chess/k3_known_test.json"),
                           load("chess/surrogate_learning_curve.json"), load("chess/calculation_features.json"))
    if atlas:
        sites = sorted(atlas["sites"], key=int)
        pick = ["material", "leela_value_q", "mobility_us", "king_zone_attacks_on_them", "passed_us", "hanging_them",
                "mate_in_1_available", "space_us", "pawn_shield_us", "checks_available"]
        ch = {"atlas": {n: {"trained": [atlas["sites"][k][n].get("auc_trained", atlas["sites"][k][n]["r2_trained"]) for k in sites],
                            "control": [atlas["sites"][k][n].get("auc_control", atlas["sites"][k][n]["r2_control"]) for k in sites],
                            "metric": "auc" if "auc_trained" in atlas["sites"][sites[0]][n] else "r2"}
                        for n in pick if n in atlas["sites"][sites[0]]}}
        ch["k3"] = k3
        ch["learning_curve"] = lc
        ch["calc"] = calc
        cands = load("chess/dark/candidates.json", [])
        ch["candidates"] = [{k: c.get(k) for k in ("round", "cluster", "name", "idea", "code", "error")} |
                            ({"accepted": c["test"]["accepted"], "criteria": c["test"]["criteria"],
                              "gain": c["test"]["gain_LD2"]["gain"], "ci": c["test"]["gain_LD2"]["ci95"],
                              "gain2": c["test"]["gain_744706"]["gain"],
                              "dec": [c["test"]["decodability"]["trained"], c["test"]["decodability"]["control"]],
                              "maxcorr": c["test"]["max_corr_library"]} if c.get("test") else {}) for c in cands]
        st = load("chess/dark/state.json")
        if st:
            meta = os.path.expanduser("~/kdata/chess/LD2/meta.json")
            ch["clusters"] = [{k: c[k] for k in ("cluster", "n", "mean_residual", "consistency", "mean_q", "mean_qhat",
                                                  "concept_z", "top_move_share", "themes")} |
                              {"prototypes": chess_boards(meta, c["prototypes"][:4]) if os.path.exists(meta) else []}
                              for c in st["clusters"][:8]]
        ch["unit_rules"] = load("chess/units/rules.json", [])
        ch["behaviour"] = load("chess/behaviour.json")
        ch["rook_pawn_context"] = load("chess/rook_pawn_context.json")
        ch["channels"] = load("chess/channel_evidence.json")
        # K10/K11: the strong network (t1-256x10) against 32,000 human decisions
        cf = load("chess/rules/confirm_fresh.json") or {}
        qt = load("chess/rules/queen_trade_fresh.json") or {}
        po = load("chess/rules/real_outcomes_pooled.json") or {}
        rf = load("chess/rules/real_outcomes_fresh.json") or {}
        if cf and qt:
            def rule(name, detail, conf, out_pooled=None, out_fresh=None, extra=""):
                o = po.get(out_pooled) if out_pooled else None
                fr = rf.get(out_fresh) if out_fresh else None
                t = {"transfer": o["adjusted_diff"], "transfer_ci": o["ci95"], "transfer_n": o["n"], "transfer_scope": "both samples"} if o else \
                    ({"transfer": fr["diff_adjusted_for_t1_q"], "transfer_ci": fr["adjusted_ci95"], "transfer_n": fr["n_followed"] + fr["n_not"],
                      "transfer_scope": "fresh games"} if fr else {})
                return {"name": name, "detail": detail, "t1": conf["t1"], "human": conf["human"], "LD2": conf["LD2"],
                        "744706": conf["744706"], "z": conf["interaction_z"], "bands": conf["bands"], "note": extra, **t}
            ch["strong"] = {
                "check": load("chess/t1_check.json"),
                "rules": [
                    rule("Quiet position → pawn lever", "a pawn push that attacks an enemy pawn", cf["c0"], "c0 quiet -> pawn lever"),
                    rule("Enemy king still in the centre → pawn lever", "strike at the pawns in front of an uncastled king", cf["c3"],
                         "c3 enemy king in centre -> pawn lever", extra="partly causal"),
                    rule("Quiet position → pawn sacrifice", "push a pawn where it can be taken for free", cf["c2"], None, "c2 quiet -> pawn sacrifice"),
                    rule("Behind in material → offer a trade", "all of the excess is in lost positions", cf["b1"], None, "b1 behind -> trade offer"),
                    rule("Lost → trade queens", "the textbook says the opposite", qt["e1"], "e1 lost -> queen trade"),
                ],
                "mechanism": qt.get("mechanism"),
                "king_cause": load("chess/rules/king_cause_fresh.json"),
                "lost_cause": load("chess/rules/lost_cause_fresh.json"),
                "trade_cause": (load("chess/rules/trade_when_behind_fresh.json") or {}).get("cause"),
                "examples": load("chess/rules/examples_fresh.json"),
                "textbook": [f"{r['condition']} → {r['move_type']}" for r in (load("chess/rules/mined_b.json") or {}).get("passing", [])[:6]],
            }
        s["chess"] = ch
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    with open(os.path.join(R, "summary.json"), "w") as f:
        json.dump(clean(s), f, allow_nan=False)
    print("summary:", {k: len(json.dumps(v)) for k, v in s.items()})


if __name__ == "__main__":
    main()
