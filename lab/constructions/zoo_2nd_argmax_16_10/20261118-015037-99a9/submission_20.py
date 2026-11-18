import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs. It runs a two-slot scan: a LEVEL (the largest "
  "value so far, with its position) and an ANSWER (the second largest, with its position). At each step x_t is compared with the stored "
  "level; if it is above, the old level drops into the answer slot and x_t becomes the new level; otherwise x_t is compared with the "
  "stored answer and replaces it if larger. Ties go to the later position. The stored slot values are not refreshed, they leak "
  "multiplicatively (the level by about 0.1 percent per step, the answer by about 0.5 percent per step), so each comparison is made "
  "against a slightly stale value: a value within about 0.1 percent per elapsed step of the level counts as ABOVE it (this demotes a "
  "near-tied maximum into the answer slot, which is why the network sometimes answers with the maximum's own position) and a value "
  "within about 0.5 percent per step of the answer replaces it (which is why it sometimes answers with the 3rd largest when the 2nd and "
  "3rd are nearly equal). Measured: the demotion boundary is the same RELATIVE fraction for input scale 0.3, 1 and 4, because the "
  "network is positively homogeneous, and the answer slot needs a wider relative window than the level slot. Stage 1: W_x has only five "
  "significant entries (units 1,6,7,8 = -10.6,-11.0,-13.2,-12.3 and unit 4 = +10.2, all others |Wx|<=0.3), so h1 = relu(x1)*a + "
  "relu(-x1)*b: below-level (negative gain) and above-level (positive gain) detectors of the single scalar x1, i.e. the two slots of a "
  "one-element scan. W_o is a matched filter over the smooth position code of the answer slot: its class vectors are shifted copies of a "
  "smooth template (adjacent-class cosines 0.4-0.9, classes 3-4 apart about -0.6), so the output decays smoothly away from the answer "
  "class, and the level's position code is still present in the state as a weaker second bump, which is what wins the near ties.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u = iv("x1_pos", np.maximum(x[:, 0], 0.0))      # relu(x1): drives the below-level detectors
    v = iv("x1_neg", np.maximum(-x[:, 0], 0.0))     # relu(-x1): drives the above-level detector
    s = x.copy(); s[:, 0] = u - v
    B, n = s.shape
    rl = 0.9990                                       # per-step leak of the stored level value
    ra = 0.9950                                       # per-step leak of the stored answer value
    Lv = np.full(B, -1e6); Lp = np.zeros(B)           # level value / position
    Av = np.full(B, -1e6); Ap = np.zeros(B)           # answer value / position
    for t in range(n):
        Lv = Lv * rl; Av = Av * ra
        xt = s[:, t]
        above = xt > Lv                               # above the (stale) level -> demote it into the answer
        sec = xt > Av
        nAv = np.where(above, Lv, np.where(sec, xt, Av))
        nAp = np.where(above, Lp, np.where(sec, float(t), Ap))
        Lv = np.where(above, xt, Lv)
        Lp = np.where(above, float(t), Lp)
        Av = nAv; Ap = nAp
    pos = iv("anspos", Ap)
    lev = iv("levpos", Lp)
    d = np.abs(np.arange(n)[None, :] - pos[:, None])
    score = -(d + 0.15 * d * d) + 1e-6 * np.arange(n)[None, :]
    dl = np.abs(np.arange(n)[None, :] - lev[:, None])
    score = score + 0.30 * np.exp(-dl * dl / 30.0)    # weaker second bump at the level's position
    return score
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
