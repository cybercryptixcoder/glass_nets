import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs (ties go to the LAST occurrence: two equal "
  "large values -> the later position wins; a value must strictly exceed the current level to demote it). "
  "Stage 1 is two-dimensional: Wx has only five significant entries (units 1,6,7,8 = -10.6,-11.0,-13.2,-12.3 and unit 4 = "
  "+10.2; every other |Wx| <= 0.3), so h1 = relu(x1)*pos + relu(-x1)*neg is a bank of below-the-level (negative gain) and "
  "above-the-level (positive gain) detectors of the single scalar x1. Each later step re-reads one new input through the "
  "same detector bank (units 1,4,6,7,8 are the only units with a significant input gain) and compares it with the state, "
  "so the state is, step by step, a linear function of the positive and negative parts of the inputs read so far "
  "(span{relu(x_s), relu(-x_s)}: 98% of h2, 90% of h3, 83% of h4, ... 69% of h10). The state maintains the level of the "
  "prefix, the position of the level holder, the current answer value and its position, and a per-step position counter "
  "(+1 per step). Each step compares x_t with the level (above -> the old level becomes the answer, x_t becomes the level) "
  "and otherwise with the answer value (above -> x_t becomes the answer). W_o is a matched filter over the smooth position "
  "code: its 10 class vectors are shifted copies of a smooth template (adjacent classes have cosine 0.4-0.9, classes 3-4 "
  "apart about -0.6), so the output decays smoothly away from the answer class and carries a large class-9 penalty. The "
  "network is positively homogeneous, so all its comparisons are relative, which is why the decision is scale invariant "
  "and why it departs from an exact second max mostly on near ties.")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u1 = iv("x1_pos", np.maximum(x[:, 0], 0.0)); v1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    u2 = iv("x2_pos", np.maximum(x[:, 1], 0.0)); v2 = iv("x2_neg", np.maximum(-x[:, 1], 0.0))
    u3 = iv("x3_pos", np.maximum(x[:, 2], 0.0)); v3 = iv("x3_neg", np.maximum(-x[:, 2], 0.0))
    s = x.copy()
    s[:, 0] = u1 - v1            # x1 recovered from its two detector groups
    s[:, 1] = u2 - v2            # x2 recovered likewise
    s[:, 2] = u3 - v3
    n = s.shape[1]
    others = np.max(s[:, None, :] - 1e4 * np.eye(n)[None], axis=2)   # max over k != j
    score = np.minimum(s, others) - 1e3 * (s > others)               # second-largest entry wins
    return score + 1e-9 * np.arange(n)[None, :]                      # ties -> later position
ALIGN = {"x1_pos": ["h1", "h2", "h3"], "x1_neg": ["h1", "h2", "h3"],
         "x2_pos": ["h2", "h3"], "x2_neg": ["h2", "h3"],
         "x3_pos": ["h3"], "x3_neg": ["h3"]}
