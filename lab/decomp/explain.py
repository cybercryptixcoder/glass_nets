"""D1: one command, one network, the whole story. Blind: reads lab/models/<id>/weights.npz only.

  python3 lab/decomp/explain.py <model_id>

Runs the group decompiler (lab/decomp/decompile.py: abelian, then non-abelian orbit mode) and factor mode
(lab/decomp/factorize.py), writes both reports to results/decomp/, and prints what the network computes in plain words.
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lab"))
from decomp import decompile as dc, factorize as fz


def main():
    mid = sys.argv[1]
    quiet = lambda s: None
    out = os.path.join(ROOT, "results", "decomp"); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    print(f"== {mid}: reading the weights (nothing else) ...")
    rep, src = dc.decompile_any(mid, 2000, quiet)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(out, mid + ".json"), "w"), indent=1)
    if src:
        open(os.path.join(out, mid + ".py"), "w").write(src + "\n")
    t1 = time.time()
    fr = fz.run(mid, quiet); fr["seconds"] = round(time.time() - t1, 1)
    json.dump(fr, open(os.path.join(out, mid + "_factor.json"), "w"), indent=1)
    print()
    v = rep.get("verdict", "")
    if v.startswith("PASS"):
        if rep.get("group_name") or str(rep.get("group", "")).startswith("non-abelian"):
            print(f"* The inputs are elements of a non-abelian group of order {rep['n_symbols']}" + (f": {rep['group_name']}." if rep.get("group_name") else "."))
        else:
            G = " x ".join(f"Z_{d}" for d in rep["group"])
            print(f"* The inputs are elements of the group {G} and the network adds them: c = CLASS_OF[g(a) + g(b)].")
            nm = rep.get("a_side", {}).get("name")
            print(f"  The symbol map is {nm}." if nm else "  The symbol map has no simple closed form: it is a relabelling, recovered as a table.")
        print(f"  Checked: the program gives the network's answer on {rep['M1']['agreement']:.1%} of all inputs; moving one input's "
              f"coordinates inside the network moves its answer as predicted in {rep['M2']['success']:.1%} of trials (random subspace: {rep['M2_random_control']['success']:.1%}).")
        print(f"  Program: results/decomp/{mid}.py")
    else:
        print(f"* No group law: {v}.")
    fv = fr.get("verdict", "")
    if fv.startswith("SPLITS"):
        print(f"* The network splits into {len(fr['parts'])} independent sub-computations:")
        for p in fr["parts"]:
            m = ", ".join(p["fingerprint"]["matches"]) or "unnamed table"
            core = p["fingerprint"].get("invertible_core", {})
            print(f"  - {p['a_blocks']} x {p['b_blocks']} blocks -> {p['values']} values: {m}" + (f" (invertible core: {core.get('group')})" if core.get("group") else ""))
        st = fr["split_test"]
        print(f"  Checked: the sum of the part tables ({st['table_entries']} entries) gives the network's answer on {st['argmax_agreement']:.1%} of {st['inputs']} inputs.")
        if fr["absorbing"]["a"] or fr["absorbing"]["b"]:
            print(f"  Set aside (their answer never depends on the other input): a = {fr['absorbing']['a']}, b = {fr['absorbing']['b']}.")
    else:
        print(f"* No split into independent parts ({fv}).")
    print(f"\n({round(time.time() - t0)} s; reports in results/decomp/; catalogue: python3 lab/server.py, then /catalog?t={mid})")


if __name__ == "__main__":
    main()
