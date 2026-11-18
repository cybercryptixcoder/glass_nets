import numpy as np

DESCRIPTION = (
    "Stage 1 (h1) is derived analytically: h1 = relu(x1 * W_x), so every unit of h1 is a "
    "nonnegative multiple of max(x1,0) (if its input weight is positive) or of max(-x1,0) "
    "(if negative). Naming u = max(x1,0) and v = max(-x1,0) therefore spans h1 exactly "
    "(h1 = u * W_x^+ + v * W_x^-), with no intercept. Any edit of (u,v) at h1 corresponds "
    "exactly to feeding the network the value x1 = u - v, so model() reconstructs x1 from the "
    "two variables iv returns and then answers the task. The task's answer is obtained by "
    "scoring every position i by -(rank_i - 1), where rank_i is the number of sequence values "
    "strictly larger than x_i: the second-largest value is the unique position with exactly "
    "one larger value, so it gets the top score (0), while the maximum (rank 0) and all lower "
    "ranks (>=2) get strictly lower scores. Thus the largest output is the position of the "
    "second-largest number, which is what the network reports. Only the first stage is "
    "constructed from first principles here; later hidden states are nonlinear in the prefix "
    "and are not claimed."
)

def model(x, iv):
    x = np.asarray(x, dtype=np.float64)
    u = iv("x1_pos", np.maximum(x[:, 0], 0.0))   # max(x1, 0)
    v = iv("x1_neg", np.maximum(-x[:, 0], 0.0))  # max(-x1, 0)
    x1 = u - v                                   # x1 recovered from the aligned variables
    vals = np.array(x, dtype=np.float64, copy=True)
    vals[:, 0] = x1                              # all later computation uses the iv values
    # rank[i] = number of positions j with vals[j] > vals[i]
    cmp = vals[:, None, :] > vals[:, :, None]    # [B, i, j] = vals[j] > vals[i]
    rank = cmp.sum(axis=2).astype(np.float64)
    return -(np.abs(rank - 1.0))                 # top score at the second-largest position

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
