"""U15: U9's data-driven unit search on argmedian h = 16 vs 32 (pre-registered in DIARY)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT / "lab"))
args = sys.argv[1:]; sys.argv = sys.argv[:1]
import unitmap
from u9_search import run
out = {}
for n in [int(a) for a in args[0].split(",")]:
    for h in (16, 32):
        for seed in range(3):
            tid = f"zoo_argmedian_{h}_{n}_s{seed}"
            m = unitmap.unit_map(tid, confidence=False); lf = {(c["t"], c["unit"]): c["fit"] for c in m["cells"]}
            later = [c for c in m["cells"] if c["t"] > 1]; tot = sum(c["var"] for c in later)
            lib = sum(c["var"] for c in later if c["fit"] >= 0.9)
            r = run(tid, "original", lambda t, u: (lf.get((t, u)) if lf.get((t, u)) is not None else 0) < 0.9)
            srch = sum(c["var"] for c in r["cells"] if c["r2_test"] >= 0.9)
            out[tid] = {"h": h, "n": n, "explained_share": r["explained_share"], "library_share": lib / tot, "total_share": (lib + srch) / tot, "n_unexplained": r["n"]}
            print(f"{tid}: library {lib/tot:.2f} | search explains {r['explained_share']:.2f} of the rest | total {(lib+srch)/tot:.2f}", flush=True)
(ROOT / f"results/glass/units/u15_{args[0].replace(',', '-')}.json").write_text(json.dumps(out, indent=1))
