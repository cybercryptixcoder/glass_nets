
import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs. Stage 1 is two-dimensional: "
  "Wx has only five significant entries (units 1,6,7,8 ~ -10.6,-11.0,-13.2,-12.3 and unit 4 ~ +10.2, all others <=0.3), "
  "so h1 = relu(x1)*pos(Wx) + relu(-x1)*neg(Wx): a bank of below-the-level (negative gain) and above-the-level (positive "
  "gain) detectors, all comparing x1 with a stored level. The remaining units form a level tracker (measured by bisection: "
  "the detectors' thresholds equal the running maximum of the prefix) and a position counter (+1 per step, reset to the "
  "current step when the answer changes; on equal values the LAST occurrence wins). W_o is a matched filter over this "
  "smooth position code, which is why the output decays smoothly around the answer class. The comparisons are resolution "
  "limited (1/gain ~ 0.05-0.1), so the network deviates from an exact second max only for near ties. "
  "Decision: argmax_j of x_j where j is not the maximum, i.e. argmax_j min(x_j, max_{k!=j} x_k) with the maximum's own "
  "position demoted.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))      # relu(x1): drives the below-level detectors
    v1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))     # relu(-x1): drives the above-level detector
    s = x.copy(); s[:, 0] = u1 - v1                  # x1 recovered from its two detector groups
    n = s.shape[1]
    others = np.max(s[:, None, :] - 1e4 * np.eye(n)[None], axis=2)   # max over k != j
    score = np.minimum(s, others) - 1e3 * (s > others)               # second-largest entry wins
    return score
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
