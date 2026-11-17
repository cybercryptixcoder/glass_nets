import numpy as np

DESCRIPTION = (
    "Task: '<m1> <year> - <m2>' -> which year the date range ends in: 'same year' or 'next year' "
    "('previous year' is never chosen). The whole decision is made by the time the final residual "
    "stream is reached. At the last token (resid12@end), the only position that feeds the "
    "unembedding, the model holds two things: (1) the identity of the year token, which selects "
    "which three vocabulary rows are the answer tokens, and (2) a per-year 'wrap' score, the "
    "next-year-minus-same-year logit difference, carried in the direction that is the difference of "
    "the 'next-year' and 'same-year' unembedding rows weighted by the final LayerNorm gain (the "
    "direction therefore differs for every year). That score is positive when the range wraps past "
    "December (m2 <= m1) and grows with m1-m2. The answer is 'next year' iff the score is positive; "
    "the 'previous year' logit stays far below both. Scores for years other than the prompt's own "
    "are near zero (the named subspace is a per-year mask of the same quantity)."
)

# mean wrap score (next-year minus same-year logit difference) per (m1, m2), averaged over years
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
    sc = TAB[m1, m2]
    wrap = np.zeros((len(x), 20))
    wrap[np.arange(len(x)), y] = sc                # per-year wrap score
    wrap = iv("wrap", wrap)

    d = wrap.sum(axis=1)                           # the active year's score
    same = np.zeros(len(x))
    prv = -12.0 * np.ones(len(x))
    return np.stack([same, d, prv], axis=1)


ALIGN = {"year": ["resid12@end"], "wrap": ["resid12@end"]}
