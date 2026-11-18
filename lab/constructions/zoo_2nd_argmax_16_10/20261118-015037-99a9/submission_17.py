import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs. It runs the classic two-slot scan: it keeps a "
  "LEVEL (the largest value seen so far) with its position and an ANSWER (the second largest) with its position. At each step x_t is "
  "compared with the stored level; if it is above, the old level drops into the answer slot and x_t becomes the new level; otherwise "
  "x_t is compared with the stored answer and replaces it if larger. Ties go to the later position. The stored slot VALUES leak very "
  "slightly (about 0.3 percent per step, factor 0.997), so a value that is within ~0.3 percent per elapsed step of the level counts as "
  "above it: that is exactly why the network demotes a near-tied maximum and answers with the maximum's own position, and why it "
  "occasionally answers with the 3rd largest when the 2nd and 3rd are nearly equal. The whole network is positively homogeneous, so the "
  "leak is a relative (multiplicative) one, which is why the decisions are invariant to rescaling the whole input and why the deviations "
  "only happen on near ties. Stage 1: W_x has only five significant entries (units 1,6,7,8 = -10.6,-11.0,-13.2,-12.3 and unit 4 = "
  "+10.2, all others |w|<=0.3), so h1 = relu(x1)*a + relu(-x1)*b: a bank of below-level (negative gain) and above-level (positive gain) "
  "detectors of the single scalar x1, i.e. the two slots of a one-element scan. W_o is a matched filter over the smooth position code of "
  "the answer slot (its class vectors are shifted copies of a smooth template, adjacent-class cosines 0.4-0.9), so the output decays "
  "smoothly away from the answer class.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u = iv("x1_pos", np.maximum(x[:, 0], 0.0))      # relu(x1): drives the below-level detectors
    v = iv("x1_neg", np.maximum(-x[:, 0], 0.0))     # relu(-x1): drives the above-level detector
    s = x.copy(); s[:, 0] = u - v
    B, n = s.shape
    r = 0.997                                        # per-step leak of the stored slot values
    Lv = np.full(B, -1e6); Lp = np.zeros(B)          # level value / position
    Av = np.full(B, -1e6); Ap = np.zeros(B)          # answer value / position
    for t in range(n):
        Lv = Lv * r; Av = Av * r                     # both slots leak
        xt = s[:, t]
        above = xt > Lv                              # above the level -> demote the level into the answer
        sec = xt > Av
        nAv = np.where(above, Lv, np.where(sec, xt, Av))
        nAp = np.where(above, Lp, np.where(sec, float(t), Ap))
        Lv = np.where(above, xt, Lv)
        Lp = np.where(above, float(t), Lp)
        Av = nAv; Ap = nAp
    pos = iv("anspos", Ap)
    score = -np.abs(np.arange(n)[None, :] - pos[:, None]) + 1e-6 * np.arange(n)[None, :]
    return score
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
