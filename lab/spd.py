"""Stochastic Parameter Decomposition engine (CHANGES A1; moved unchanged from scripts/glass/units/spd_core.py). After Bushnaq, Braun & Sharkey 2025, arXiv 2506.20790, written
for small networks given as a forward function over named weight matrices. Pre-registered in docs/glass/DIARY.md.

Each target matrix W [d_out, d_in] is decomposed as sum_c U_c V_c^T (C rank-1 subcomponents). A small MLP per
subcomponent reads its inner activation V_c . a (a = the matrix's input) and gives a causal importance g_c in [0, 1].
Training minimises
  faithfulness            ||W - sum_c U_c V_c^T||^2
  stochastic reconstruction  || f(x; masked weights) - f(x; W) ||^2, masks m_c = g_c + (1 - g_c) r, r ~ U(0, 1)
                             (all matrices masked at once, and one matrix at a time)
  importance minimality   sum_c |g_c|^p
The forward function is written once, against a `mm(name, a)` callback that applies matrix `name` to `a`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lower_leaky(z):          # used for masks: gradient flows below 0
    return torch.where(z < 0, 0.01 * z, z.clamp(max=1.0))


def upper_leaky(z):          # used for the minimality penalty: gradient flows above 1
    return torch.where(z > 1, 1 + 0.01 * (z - 1), z.clamp(min=0.0))


class Decomposed(nn.Module):
    def __init__(self, W, C, hidden=16):
        super().__init__()
        d_out, d_in = W.shape
        self.register_buffer("W", W.clone())
        # initialisation as in the SPD code release (init_As_and_Bs_): unit-norm random U_c and V_c, then U_c scaled
        # by the overlap U_c . W V_c, so that each subcomponent starts as W's projection on a random rank-1 direction
        U = torch.randn(C, d_out, dtype=W.dtype); U = U / U.norm(dim=-1, keepdim=True)
        V = torch.randn(C, d_in, dtype=W.dtype); V = V / V.norm(dim=-1, keepdim=True)
        U = U * torch.einsum("co,oi,ci->c", U, W, V)[:, None]
        self.U = nn.Parameter(U); self.V = nn.Parameter(V)
        self.w1 = nn.Parameter(torch.randn(C, hidden) * math.sqrt(2.0)); self.b1 = nn.Parameter(torch.zeros(C, hidden))
        self.w2 = nn.Parameter(torch.randn(C, hidden) / math.sqrt(hidden)); self.b2 = nn.Parameter(torch.zeros(C))

    def ci_pre(self, a):                       # a: [B, d_in] -> pre-sigmoid importance [B, C]
        inner = a @ self.V.T
        return (F.gelu(inner[..., None] * self.w1 + self.b1) * self.w2).sum(-1) + self.b2

    def apply(self, a, mask=None):             # masked component product, or the target matrix when mask is None
        if mask is None:
            return a @ self.W.T
        return ((a @ self.V.T) * mask) @ self.U

    def weight(self):
        return self.U.T @ self.V                # [d_out, d_in]


def _run(forward, x, mods, mask_for, record=None):
    """One forward pass where the k-th call of matrix `name` uses mask_for(name, k) (None = the target matrix).
    A matrix may be called several times per pass (a recurrent W_h): each call gets its own importances and masks."""
    count = {}

    def mm(name, a):
        k = count.get(name, 0); count[name] = k + 1
        if record is not None:
            record.setdefault(name, []).append(a)
        return mods[name].apply(a, mask_for(name, k))
    return forward(x, mm)


def decompose(forward, weights, sample, C=20, steps=20000, batch=2048, lr=2e-3, coef_faith=1e3, coef_rec=1.0,
              coef_min=3e-3, p=1.0, seed=0, log=500, device="cpu", dtype=torch.float64, recon="mse", kl_scale=1.0, coef_ci=0.0,
              ci_mode="soft"):
    """forward(x, mm) -> output, where mm(name, a) applies matrix `name` to a; weights: {name: tensor [d_out, d_in]};
    sample(n) -> a batch of inputs. Returns the trained {name: Decomposed}."""
    torch.manual_seed(seed)
    mods = nn.ModuleDict({k: Decomposed(v.double(), C).double() for k, v in weights.items()}).to(device=device, dtype=dtype)
    _sample = sample
    sample = lambda n: _sample(n).to(device=device, dtype=dtype)        # noqa: E731  (CHANGES A6, A9)
    opt = torch.optim.Adam(mods.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    for step in range(steps):
        x = sample(batch); acts = {}
        with torch.no_grad():
            y_t = _run(forward, x, mods, lambda n, k: None, record=acts)
        gs = {k: [mods[k].ci_pre(a) for a in acts[k]] for k in mods}               # per call
        masks = {k: [lower_leaky(z) + (1 - lower_leaky(z)) * torch.rand_like(z) for z in zs] for k, zs in gs.items()}
        if recon == "kl":   # A14: decision-focused reconstruction (softmax KL on the original logit scale)
            lt = F.log_softmax(y_t * kl_scale, -1)
            loss_fn = lambda y: F.kl_div(F.log_softmax(y * kl_scale, -1), lt, log_target=True, reduction="batchmean")  # noqa: E731
        else:
            loss_fn = lambda y: F.mse_loss(y, y_t)  # noqa: E731
        rec = loss_fn(_run(forward, x, mods, lambda n, k: masks[n][k]))
        for name in mods:   # one matrix at a time
            rec = rec + loss_fn(_run(forward, x, mods, lambda n, k, name=name: masks[n][k] if n == name else None)) / len(mods)
        if coef_ci:         # A38: CI-masked reconstruction (masks = importances, r = 0): every unimportant component off at once
            def _cm(z):         # A38d: mask VALUE = the importance the atlas uses (clamped to [0, 1]); gradient = the leaky one
                g = lower_leaky(z)
                return z.clamp(0, 1) + g - g.detach()
            def _hm2(z):        # on/off mask 1[z > 0.5], gradient = the leaky one (straight-through)
                g = lower_leaky(z)
                return (z > 0.5).to(z.dtype) + g - g.detach()
            if ci_mode == "layer_hard":   # A38e (experiment): one matrix at a time, on/off masks, the rest exact
                pick = list(mods)[int(torch.randint(len(mods), (1,)))]
                rec_ci = loss_fn(_run(forward, x, mods, lambda n, k: _hm2(gs[n][k]) if n == pick else None))
            elif ci_mode == "layer":  # A38c (experiment): one matrix at a time (a random one per step), masks = g, the rest exact
                pick = list(mods)[int(torch.randint(len(mods), (1,)))]
                rec_ci = loss_fn(_run(forward, x, mods, lambda n, k: _cm(gs[n][k]) if n == pick else None))
            elif ci_mode == "hard":   # A38b (experiment): on/off masks 1[g > 0.5], straight-through gradient to g
                def _hm(z):
                    g = lower_leaky(z)
                    return (z > 0.5).to(z.dtype) + g - g.detach()
                rec_ci = loss_fn(_run(forward, x, mods, lambda n, k: _hm(gs[n][k])))
            else:
                rec_ci = loss_fn(_run(forward, x, mods, lambda n, k: _cm(gs[n][k])))
            rec = rec + coef_ci * rec_ci
        faith = sum(((m.W - m.weight()) ** 2).mean() for m in mods.values())
        mini = sum((upper_leaky(torch.stack(zs)) ** p).sum(-1).mean() for zs in gs.values())
        loss = coef_faith * faith + coef_rec * rec + coef_min * mini
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and step % log == 0:
            alive = {k: int((upper_leaky(torch.stack(zs)).mean((0, 1)) > 0.01).sum()) for k, zs in gs.items()}
            print(f"step {step:6d} loss {loss.item():.4f} faith {faith.item():.2e} rec {rec.item():.2e} min {mini.item():.3f} alive {alive}", flush=True)
    return mods


def importance(mods, forward, x):
    """Causal importances (clamped to [0, 1]) of every subcomponent on inputs x: {name: [calls, B, C]}
    (calls = 1 for a matrix used once per forward pass)."""
    acts = {}
    with torch.no_grad():
        _run(forward, x, mods, lambda n, k: None, record=acts)
        return {k: torch.stack([mods[k].ci_pre(a) for a in acts[k]]).clamp(0, 1) for k in mods}
