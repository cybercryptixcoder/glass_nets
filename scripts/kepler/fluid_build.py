"""Build the fluid target: generate videos of the flow, then train the video model on them.

    python3 scripts/kepler/fluid_build.py gen     # data -> $KDATA/fluid/*.npy, sealed parameters
    python3 scripts/kepler/fluid_build.py train   # model -> $KDATA/fluid/ckpt_*.pt (and results/kepler/fluid/model.pt)

$KDATA defaults to ~/kdata (outside the repo: the videos are ~250 MB).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from kepler.targets.fluid.sim import Flow, seal  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
OUT = os.path.join(KDATA, "fluid")
FRAME_DT = 0.05
REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def gen():
    os.makedirs(OUT, exist_ok=True)
    flow = Flow()                       # the hidden law (defaults in sim.py)
    seal(flow.params(), os.path.join(OUT, "sealed_params.json"))
    rng = np.random.default_rng(20261124)
    parts = []
    for spin in [0, 0, 1, 2, 4, 6, 10, 15, 20, 30]:
        parts.append(flow.trajectories(rng, batch=32, frames=40, frame_dt=FRAME_DT, spinup=spin))
        log(f"train chunk spinup={spin} rms={parts[-1].std():.3f}")
    np.save(os.path.join(OUT, "train.npy"), np.concatenate(parts))
    rng = np.random.default_rng(777)
    val = [flow.trajectories(rng, batch=16, frames=40, frame_dt=FRAME_DT, spinup=s) for s in [0, 5, 20, 40]]
    np.save(os.path.join(OUT, "val.npy"), np.concatenate(val))
    # Out of distribution: stronger and finer fields than any training start (law vs. lookup)
    rng = np.random.default_rng(999)
    sp_init = flow.random_init
    flow.random_init = lambda r, b, amp_range=(6.0, 9.0), k_peak_range=(9.0, 13.0): sp_init(r, b, amp_range, k_peak_range)
    np.save(os.path.join(OUT, "ood.npy"), flow.trajectories(rng, batch=32, frames=12, frame_dt=FRAME_DT))
    log("done gen")


def train(steps=8000, bs=8, width=32, lr=2e-3, seed=0):
    import torch
    torch.set_num_threads(2)
    from kepler.targets.fluid.model import FluidNet, save
    torch.manual_seed(seed)
    tr = np.load(os.path.join(OUT, "train.npy"), mmap_mode="r")
    va = np.load(os.path.join(OUT, "val.npy"))
    n_traj, n_fr = tr.shape[:2]
    idx = [(i, t) for i in range(n_traj) for t in range(n_fr - 1)]
    sample = np.asarray(tr[::7, ::5])
    in_scale = float(sample.std())
    out_scale = float(np.asarray(tr[::7, 1:] - tr[::7, :-1]).std())
    m = FluidNet(width)
    m.in_scale.fill_(in_scale)
    m.out_scale.fill_(out_scale)
    torch.manual_seed(1234)
    rand = FluidNet(width)              # control: same architecture, never trained
    rand.in_scale.fill_(in_scale)
    rand.out_scale.fill_(out_scale)
    save(rand, os.path.join(OUT, "ckpt_random.pt"), {"steps": 0, "control": True})
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    vx = torch.tensor(va[:, :-1].reshape(-1, 64, 64)[::10][:256])
    vy = torch.tensor(va[:, 1:].reshape(-1, 64, 64)[::10][:256])
    t0 = time.time()
    for step in range(1, steps + 1):
        pick = rng.integers(0, len(idx), bs)
        x = torch.tensor(np.stack([tr[idx[k][0], idx[k][1]] for k in pick]))
        y = torch.tensor(np.stack([tr[idx[k][0], idx[k][1] + 1] for k in pick]))
        loss = (((m(x) - y) / out_scale) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                pv = m(vx)
                rel = float(((pv - vy) ** 2).sum().sqrt() / ((vy - vx) ** 2).sum().sqrt())
            log(f"step {step} loss {loss.item():.4f} val_rel_err_of_change {rel:.4f} ({time.time()-t0:.0f}s)")
        if step in (1000, 2000, 4000) or step == steps:
            save(m, os.path.join(OUT, f"ckpt_{step}.pt"), {"steps": step, "frame_dt": FRAME_DT})
    os.makedirs(os.path.join(REPO, "results", "kepler", "fluid"), exist_ok=True)
    save(m, os.path.join(REPO, "results", "kepler", "fluid", "model.pt"), {"steps": steps, "frame_dt": FRAME_DT})
    log("done train")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen()
    elif cmd == "train":
        train(steps=int(os.environ.get("STEPS", 8000)))
