import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs (ties go to the LAST occurrence: two equal "
  "large values -> the later position wins; a value must strictly exceed the current level to demote it). "
  "Stage 1 is two-dimensional: Wx has only five significant entries (units 1,6,7,8 = -10.6,-11.0,-13.2,-12.3 and unit 4 = "
  "+10.2; every other |Wx| <= 0.3), so h1 = relu(x1)*pos + relu(-x1)*neg is a bank of below-the-level (negative gain) and "
  "above-the-level (positive gain) detectors of the single scalar x1. The state then maintains a level (the detectors' "
  "thresholds follow the running level of the prefix), the position of the level holder, the current answer value and its "
  "position, and a per-step position counter (+1 per step). Each step compares x_t with the level (above -> the old level "
  "becomes the answer, x_t becomes the level) and otherwise with the answer value (above -> x_t becomes the answer). W_o is "
  "a matched filter over the smooth position code: its 10 class vectors are shifted copies of a smooth template (adjacent "
  "classes have cosine 0.4-0.9, classes 3-4 apart about -0.6), so the output decays smoothly away from the answer class. "
  "The whole network is positively homogeneous, so all its comparisons are relative ones, which is why the decision is "
  "invariant to scaling the input and why the network departs from an exact second max mostly on near ties.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))      # relu(x1): drives the below-level detectors
    v1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))     # relu(-x1): drives the above-level detector
    s = x.copy(); s[:, 0] = u1 - v1                  # x1 recovered from its two detector groups
    n = s.shape[1]
    others = np.max(s[:, None, :] - 1e4 * np.eye(n)[None], axis=2)   # max over k != j
    score = np.minimum(s, others) - 1e3 * (s > others)               # second-largest entry wins
    return score + 1e-9 * np.arange(n)[None, :]                      # ties -> later position
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
