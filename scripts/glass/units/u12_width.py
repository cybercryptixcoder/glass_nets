"""U12: precision vs capacity at a fixed task (2nd argmax, n = 5; ARC's h = 8/16/32 checkpoints, seeds 0-4). Pre-registered in DIARY."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT / "lab"))
hs = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [8, 16, 32]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5             # U13: n = 10
TASK = sys.argv[3] if len(sys.argv) > 3 else "2nd_argmax"     # U14: argmedian
sys.argv = sys.argv[:1]                     # u10_tolerance reads sys.argv at import
import unitmap
from u8_seeds import build
from u10_tolerance import joint
out = {}
for h in hs:
    for seed in range(5):
        tmpl = ("zoo_2nd_argmax_8_5" if N == 5 else "zoo_2nd_argmax_16_10") if TASK == "2nd_argmax" else f"zoo_argmedian_16_{N}"
        tid = build(TASK, h, N, seed, tmpl)
        m = unitmap.unit_map(tid); js, share, k = joint(tid, m)
        acc = json.loads((ROOT / "lab/models" / tid / "meta.json").read_text())["test_acc"]
        out[tid] = {"h": h, "acc": acc, "eps95": m["eps95"], "coverage_history": m["coverage_history"], "coverage_raw": m["coverage_raw"],
                    "joint_substitution": js, "substituted_var_share": share, "n_substituted": k}
        print(f"{tid}: acc {acc:.3f} eps95 {m['eps95']:.4f} cov hist {m['coverage_history']:.2f} raw {m['coverage_raw']:.2f} | joint {100*js:.1f}% ({k} steps, {100*share:.0f}% var)", flush=True)
(ROOT / f"results/glass/units/u12_width_{'' if TASK == '2nd_argmax' else TASK + '_'}{'-'.join(map(str, hs))}{'' if N == 5 else f'_n{N}'}.json").write_text(json.dumps(out, indent=1))
