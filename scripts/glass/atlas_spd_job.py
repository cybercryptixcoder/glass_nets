"""Decompose targets with SPD (CHANGES A6) and save .pt files in the format lab/atlas.py loads.
    python3 scripts/glass/atlas_spd_job.py OUTDIR STEPS BATCH DEVICE target[:variant] ...   (target 'planted_rnn' for the known answer)"""
import sys, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))      # --coef-min=1e-3 --recon=kl --tag=x (A14)
out, steps, batch, device = Path(args[0]), int(args[1]), int(args[2]), args[3]
out.mkdir(parents=True, exist_ok=True)
for spec in args[4:]:
    tid, _, var = spec.partition(":"); var = var or "original"
    tgt = (atlas.planted_rnn() if tid == "planted_rnn" else atlas.planted_transformer() if tid == "planted_transformer"
           else atlas.algzoo_transformer(tid, var) if tid.startswith("lc_") else atlas.algzoo_rnn(tid, var))
    t0 = time.time()
    mods = atlas.decompose(tgt, steps=steps, batch=batch, device=device, log=2000, coef_min=float(opts.get("coef-min", 3e-3)), recon=opts.get("recon", "mse"))
    tag = ("_" + opts["tag"]) if "tag" in opts else ""
    torch.save({"mods": mods.state_dict(), "SD": tgt.out_scale, "steps": steps, "batch": batch, "device": device, "opts": opts},
               out / f"spd_{tid}_{var}_{steps}_{batch}{tag}.pt")
    print(f"DONE {spec} in {time.time() - t0:.0f}s", flush=True)
