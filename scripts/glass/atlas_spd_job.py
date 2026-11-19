"""Decompose targets with SPD (CHANGES A6) and save .pt files in the format lab/atlas.py loads.
    python3 scripts/glass/atlas_spd_job.py OUTDIR STEPS BATCH DEVICE target[:variant] ...   (target 'planted_rnn' for the known answer)"""
import sys, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
out, steps, batch, device = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[5:]:
    tid, _, var = spec.partition(":"); var = var or "original"
    tgt = atlas.planted_rnn() if tid == "planted_rnn" else atlas.algzoo_rnn(tid, var)
    t0 = time.time(); mods = atlas.decompose(tgt, steps=steps, batch=batch, device=device, log=2000)
    torch.save({"mods": mods.state_dict(), "SD": tgt.out_scale, "steps": steps, "batch": batch, "device": device},
               out / f"spd_{tid}_{var}_{steps}_{batch}.pt")
    print(f"DONE {spec} in {time.time() - t0:.0f}s", flush=True)
