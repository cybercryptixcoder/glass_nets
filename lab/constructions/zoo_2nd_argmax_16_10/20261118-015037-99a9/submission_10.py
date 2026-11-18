
import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs. Only five input weights are "
  "significant (units 1,6,7,8 with Wx ~ -10.6,-11.0,-13.2,-12.3 and unit 4 with +10.2); the rest are <=0.3. So stage 1 "
  "is two-dimensional: h1 = relu(x1)*pos(Wx) + relu(-x1)*neg(Wx), i.e. a bank of above-level / below-level detectors all "
  "driven by x1 against a stored level. The remaining 11 units are a leaky level tracker (the prefix maximum, measured by "
  "bisection: each detector's threshold equals the prefix max) plus a position counter that counts up one per step and is "
  "reset when the answer changes (impulse at step t gives the counter unit 10.16 at h10, and equal values leave the answer "
  "at the LAST occurrence). The readout W_o is a matched filter over that smooth position code. The comparison is blurred: "
  "deviations from an exact second-max occur only for gaps below ~0.05-0.1 (the detector resolution 1/gain), which is why "
  "the network errs on 4.5% of inputs. Decision = argmax_j min(x_j, max_{k!=j} x_k).")
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u1 = iv("x1_pos", np.maximum(x[:, 0], 0.0))          # relu(x1): fires the below-level detectors
    v1 = iv("x1_neg", np.maximum(-x[:, 0], 0.0))         # relu(-x1): fires the above-level detector
    s = x.copy()
    s[:, 0] = u1 - v1                                    # x1 recovered from the two detector groups
    others = np.max(s[:, None, :] * (1.0 - np.eye(s.shape[1])[None]), axis=2)   # max over k != j
    score = np.minimum(s, others)                        # = x_j if x_j is the second largest
    return score
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
