"""Smoke video model: generate dye-only videos of the flow, then train a video model on 3 past frames -> next frame.

    python3 scripts/kepler/dye_build.py gen      # $KDATA/dye/{train,val}_dye.npy, val_omega.npy, sealed_params.json
    python3 scripts/kepler/dye_build.py train    # $KDATA/dye/model.pt (GPU if available), log every 250 steps

The model never sees vorticity or velocity: only the dye frames. val_omega.npy is kept for the probes and tests.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from kepler.targets.fluid.sim import DyeFlow, seal  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
OUT = os.path.join(KDATA, "dye")
FRAME_DT = 0.05
K = 3          # past frames the model sees


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def gen(n_train=int(os.environ.get("N_TRAIN", 384)), n_val=48, frames=32):
    os.makedirs(OUT, exist_ok=True)
    flow = DyeFlow()
    seal(flow.params(), os.path.join(OUT, "sealed_params.json"))
    rng = np.random.default_rng(4242)
    tr = []
    for i in range(0, n_train, 32):
        d, _ = flow.dye_videos(rng, 32, frames, FRAME_DT, spinup=float(rng.choice([5, 10, 20, 30])))
        tr.append(d)
        log(f"train {i + 32}/{n_train}")
    np.save(os.path.join(OUT, "train_dye.npy"), np.concatenate(tr))
    rng = np.random.default_rng(99)
    vd, vo = [], []
    for i in range(0, n_val, 16):
        d, o = flow.dye_videos(rng, 16, frames, FRAME_DT, spinup=float(rng.choice([5, 10, 20, 30])))
        vd.append(d)
        vo.append(o)
    np.save(os.path.join(OUT, "val_dye.npy"), np.concatenate(vd))
    np.save(os.path.join(OUT, "val_omega.npy"), np.concatenate(vo))
    log("done gen")


def train(steps=int(os.environ.get("STEPS", 10000)), bs=32, width=int(os.environ.get("WIDTH", 64)),
          lr=float(os.environ.get("LR", 6e-4))):
    import torch
    from kepler.targets.fluid.model import FluidNet, save
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tr = torch.tensor(np.load(os.path.join(OUT, "train_dye.npy")), device=dev)     # [V, T, n, n]
    va = torch.tensor(np.load(os.path.join(OUT, "val_dye.npy")), device=dev)
    V, T = tr.shape[:2]
    in_scale = float(tr.std())
    out_scale = float((tr[:, 1:] - tr[:, :-1]).std())
    m = FluidNet(width, (1, 2, 4, 8, 16, 1, 2, 4), cin=K).to(dev)
    m.in_scale.fill_(in_scale)
    m.out_scale.fill_(out_scale)
    torch.manual_seed(1234)
    rand = FluidNet(width, (1, 2, 4, 8, 16, 1, 2, 4), cin=K)
    rand.in_scale.fill_(in_scale)
    rand.out_scale.fill_(out_scale)
    save(rand, os.path.join(OUT, "model_random.pt"), {"control": True, "k": K})
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator(device="cpu").manual_seed(0)
    vi = torch.arange(0, va.shape[0])
    vx = torch.stack([va[:, t - K + 1:t + 1] for t in range(K - 1, T - 1, 4)], 1).flatten(0, 1)
    vy = torch.stack([va[:, t + 1] for t in range(K - 1, T - 1, 4)], 1).flatten(0, 1)
    t0 = time.time()
    for step in range(1, steps + 1):
        v = torch.randint(0, V, (bs,), generator=g)
        t = torch.randint(K - 1, T - 1, (bs,), generator=g)
        x = torch.stack([tr[v, t - K + 1 + j] for j in range(K)], 1)
        y = tr[v, t + 1]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            pred = m(x)
        loss = (((pred.float() - y) / out_scale) ** 2).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0 or step == 1:
            with torch.no_grad():
                pv = torch.cat([m(vx[i:i + 256]) for i in range(0, len(vx), 256)])
                rel = float(((pv - vy) ** 2).sum().sqrt() / ((vy - vx[:, -1]) ** 2).sum().sqrt())
            log(f"step {step} loss {loss.item():.4f} val_rel_err_of_change {rel:.4f} ({time.time() - t0:.0f}s, {dev})")
        if step % 5000 == 0 or step == steps:
            save(m.cpu(), os.path.join(OUT, "model.pt"), {"steps": step, "frame_dt": FRAME_DT, "k": K})
            m.to(dev)
    log("done train")


if __name__ == "__main__":
    {"gen": gen, "train": train}[sys.argv[1]]()
