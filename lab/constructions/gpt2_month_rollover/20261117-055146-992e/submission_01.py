import numpy as np

DESCRIPTION = (
    "Date-range task: '<m1> <year> - <m2>'; the answer is the year in which the range ends: "
    "'same year' if m2 does not precede m1 in the calendar, 'next year' if the range wraps past "
    "December (m2 <= m1). The model never answers 'previous year'. "
    "I read the final residual stream at the last token (resid12@end), which is the only position "
    "that reaches the unembedding. It holds (a) the identity of the year token (which fixes which "
    "three vocabulary rows are the answer tokens) and (b) a single scalar 'wrap' score: the "
    "next-year-minus-same-year logit difference, a smooth function of (m1, m2) that is positive "
    "when m2 precedes or equals m1 and negative otherwise. Behaviour = sign of that score."
)

# mean (next-year minus same-year) logit difference measured over the 20 years, indexed [m1, m2]
TAB = np.array([
 [ 2.1,  0.1,  0.3,  0.3,  0.4,  0.4,  0.3, -0.1, -0.2, -0.4, -0.6, -0.6],
 [ 1.5,  1.8, -0.5, -0.2, -0.3,  0.3,  0.2, -0.2, -0.6, -0.5, -0.7, -0.7],
 [ 1.5,  1.1,  1.1, -0.4, -0.3, -0.1,  0.2, -0.1, -0.5, -0.6, -0.6, -0.6],
 [ 1.7,  1.4,  0.9,  1.6, -0.4,  0.1,  0.1, -0.0, -0.3, -0.4, -0.8, -0.7],
 [ 1.9,  1.4,  1.1,  0.9,  0.8, -0.2,  0.1, -0.1, -0.2, -0.5, -0.6, -0.6],
 [ 2.3,  2.0,  1.6,  1.5,  0.9,  1.6, -0.1, -0.3, -0.5, -0.2, -0.5, -0.5],
 [ 2.6,  2.2,  1.9,  1.7,  1.2,  1.4,  1.6, -0.2, -0.3, -0.4, -0.2, -0.4],
 [ 2.7,  2.4,  2.0,  2.0,  1.1,  1.5,  1.2,  1.4, -0.4, -0.2, -0.4, -0.2],
 [ 2.9,  2.4,  2.1,  2.1,  1.4,  1.6,  1.4,  1.2,  1.7, -0.4, -0.4, -0.2],
 [ 3.2,  2.8,  2.2,  2.4,  1.4,  2.0,  1.7,  1.6,  1.3,  1.6, -0.5,  0.1],
 [ 3.5,  3.0,  2.6,  2.7,  1.9,  2.3,  2.1,  1.9,  1.6,  1.0,  1.9, -0.3],
 [ 3.2,  3.0,  2.6,  2.6,  1.8,  2.4,  2.0,  1.9,  1.6,  1.1,  0.9,  1.8]])


def model(x, iv):
    x = np.asarray(x)
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    y = x[:, 2].astype(int) - 1980

    yr = iv("year", np.eye(20)[y])                 # year-token identity (categorical)
    sc = iv("wrap_score", TAB[m1, m2])             # next-year minus same-year logit difference

    same = np.zeros(len(x))
    prev = -10.0 * np.ones(len(x))
    return np.stack([same, sc, prev], axis=1)


ALIGN = {"year": ["resid12@end"], "wrap_score": ["resid12@end"]}
