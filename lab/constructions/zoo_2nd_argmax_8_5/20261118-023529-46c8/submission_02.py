
import numpy as np

DESCRIPTION = ("The network's task is the index of the SECOND LARGEST of the five inputs (agrees with 88% of its "
               "decisions; the rest are near-ties). Stage 1 is exact: h1 = x1 * wx, and wx has a fixed sign pattern, "
               "so h1 lies in the 2-D cone spanned by the 'x1>0' ray (units 0,2,4,7) and the 'x1<0' ray "
               "(units 1,3,5,6); the two variables are the positive and negative parts of x1. Later stages are "
               "provisional: they are named as the positive/negative parts of the running top-1 and top-2 values of "
               "the prefix, to be measured against the real activations. The output scores each class by how close "
               "the corresponding input is to the running second-largest value, so the largest score is the second "
               "largest input's position.")


def model(x, iv):
    B = x.shape[0]
    u1 = iv("u1", np.stack([np.maximum(x[:, 0], 0), np.maximum(-x[:, 0], 0)], 1))

    m1 = np.full(B, -1e9)
    m2 = np.full(B, -1e9)
    for t in range(5):
        xt = x[:, t]
        newm1 = np.maximum(m1, xt)
        m2 = np.where(xt > m1, m1, np.maximum(m2, xt))
        m1 = newm1
        pre = iv("pre%d" % (t + 1),
                 np.stack([np.maximum(m1, 0), np.maximum(-m1, 0), np.maximum(m2, 0), np.maximum(-m2, 0)], 1))
        _ = pre
    second = m2
    return -np.abs(x - second[:, None])


ALIGN = {"u1": ["h1"], "pre1": ["h1"], "pre2": ["h2"], "pre3": ["h3"], "pre4": ["h4"], "pre5": ["h5"]}
