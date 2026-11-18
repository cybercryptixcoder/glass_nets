import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs (ties resolved toward the LAST "
  "occurrence: with two equal large values and small others the later position wins). Stage 1 is two-dimensional: Wx has "
  "only five significant entries (units 1,6,7,8 ~ -10.6,-11.0,-13.2,-12.3 and unit 4 ~ +10.2; all others <=0.3), so "
  "h1 = relu(x1)*pos + relu(-x1)*neg is a bank of below-the-level (negative gain) and above-the-level (positive gain) "
  "detectors of the single scalar x1. Later units form a level tracker (the detectors' thresholds are a smoothed running "
  "level of the prefix) plus position counters (+1 per step). W_o is a matched filter over this smooth position code, so "
  "the output decays smoothly away from the answer class and class 9 carries a constant penalty. Because the comparisons "
  "are gain limited (~0.05-0.1 resolution) the network deviates from an exact second max only on near ties.")
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
