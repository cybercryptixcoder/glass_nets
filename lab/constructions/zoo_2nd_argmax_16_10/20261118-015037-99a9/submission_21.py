import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs. It runs a two-slot scan: a LEVEL (the largest value "
  "so far, with its position) and an ANSWER (the second largest, with its position). At each step x_t is compared with the stored level; "
  "if it exceeds it, the old level drops into the answer slot and x_t becomes the new level; otherwise x_t is compared with the stored "
  "answer. Both updates are made with HYSTERESIS: a candidate must beat the stored value by a small relative margin (about 0.2 percent "
  "for the level, about 0.3 percent for the answer) or the stored value and position are kept. So a new level needs a clear margin over "
  "the old one, and a demoted level only displaces the answer when it is clearly bigger than the answer already stored. That hysteresis "
  "is exactly why the network's decisions depart from the exact second maximum only on near ties: a near-tied maximum is demoted into the "
  "answer slot and the network answers with the maximum's own position, and a 2nd/3rd pair within the margin keeps whichever of the two "
  "reached the slot first. Measured: with a level at position q and a second value f*q at a later position, the demotion boundary is at "
  "f = 0.995 to 0.9995 depending on the gap in steps, and the same relative boundary is found for inputs scaled by 0.3, 1 and 4 - the "
  "network is positively homogeneous, so the margin is multiplicative, not absolute; a pair of exactly equal values gives the later "
  "position because the second one fills the empty answer slot. Stage 1: W_x has only five significant entries (units 1,6,7,8 = -10.6, "
  "-11.0, -13.2, -12.3 and unit 4 = +10.2, all others |Wx|<=0.3), so h1 = relu(x1)*a + relu(-x1)*b: below-level (negative gain) and "
  "above-level (positive gain) detectors of the single scalar x1, i.e. the two slots of a one-element scan. W_o is a matched filter over "
  "the smooth position code of the answer slot: its class vectors are shifted copies of a smooth template (adjacent-class cosines "
  "0.4-0.9, classes 3-4 apart about -0.6), so the output decays smoothly away from the answer class, and the level's position code is "
  "still present in the state as a weaker second bump, which is what wins the near ties.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u = iv("x1_pos", np.maximum(x[:, 0], 0.0))      # relu(x1): drives the below-level detectors
    v = iv("x1_neg", np.maximum(-x[:, 0], 0.0))     # relu(-x1): drives the above-level detector
    s = x.copy(); s[:, 0] = u - v
    B, n = s.shape
    kL, kA = 0.002, 0.003                            # hysteresis margins (relative) of the two slots
    Lv = np.full(B, -1e6); Lp = np.zeros(B)          # level value / position
    Av = np.full(B, -1e6); Ap = np.zeros(B)          # answer value / position
    for t in range(n):
        xt = s[:, t]
        above = xt > Lv * (1.0 + kL)                 # clear margin over the level -> demote the level
        cand = np.where(above, Lv, xt)               # candidate for the answer slot
        candp = np.where(above, Lp, float(t))
        upd = cand > Av * (1.0 + kA)                 # answer slot needs a clear margin too
        Av = np.where(upd, cand, Av); Ap = np.where(upd, candp, Ap)
        Lv = np.where(above, xt, Lv); Lp = np.where(above, float(t), Lp)
    pos = iv("anspos", Ap)
    lev = iv("levpos", Lp)
    d = np.abs(np.arange(n)[None, :] - pos[:, None])
    score = -(d + 0.15 * d * d) + 1e-6 * np.arange(n)[None, :]
    dl = np.abs(np.arange(n)[None, :] - lev[:, None])
    score = score + 0.30 * np.exp(-dl * dl / 30.0)    # weaker second bump at the level's position
    return score
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
