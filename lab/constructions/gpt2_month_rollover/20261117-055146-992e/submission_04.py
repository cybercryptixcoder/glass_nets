import numpy as np

DESCRIPTION = (
    "Task: '<m1> <year> - <m2>' -> which year the date range ends in: 'same year' or 'next year' "
    "('previous year' is never chosen). The decision is complete by the final residual stream. At the "
    "last token (resid12@end), the only position that feeds the unembedding, the model holds two "
    "things: (1) the identity of the year token, which selects which three vocabulary rows are the "
    "answer tokens, and (2) a per-year 'wrap' score = the next-year-minus-same-year logit difference, "
    "carried in the direction of the difference of the 'next-year' and 'same-year' unembedding rows "
    "weighted by the final LayerNorm gain (so the direction differs for every year). The score is "
    "positive when the range wraps past December (m2 <= m1) and grows with m1-m2. Answer is 'next "
    "year' iff the score is positive."
)

# next-year minus same-year logit difference, mean per (m1, m2) and per-year offset
TAB = np.array([
 [ 2.08,  0.09,  0.33,  0.31,  0.36,  0.38,  0.34, -0.06, -0.17, -0.40, -0.56, -0.62],
 [ 1.46,  1.76, -0.50, -0.16, -0.29,  0.25,  0.23, -0.15, -0.55, -0.53, -0.66, -0.69],
 [ 1.48,  1.11,  1.06, -0.43, -0.26, -0.10,  0.22, -0.12, -0.46, -0.55, -0.63, -0.62],
 [ 1.69,  1.43,  0.94,  1.61, -0.45,  0.05,  0.08, -0.01, -0.30, -0.42, -0.80, -0.68],
 [ 1.89,  1.42,  1.13,  0.89,  0.83, -0.22,  0.08, -0.14, -0.24, -0.48, -0.59, -0.65],
 [ 2.29,  1.99,  1.55,  1.49,  0.87,  1.57, -0.08, -0.28, -0.52, -0.19, -0.48, -0.48],
 [ 2.58,  2.22,  1.91,  1.74,  1.22,  1.42,  1.60, -0.25, -0.31, -0.38, -0.21, -0.40],
 [ 2.71,  2.42,  1.98,  1.95,  1.05,  1.53,  1.24,  1.44, -0.43, -0.16, -0.40, -0.16],
 [ 2.90,  2.40,  2.10,  2.10,  1.37,  1.64,  1.42,  1.17,  1.65, -0.42, -0.36, -0.18],
 [ 3.22,  2.77,  2.22,  2.37,  1.41,  2.00,  1.67,  1.61,  1.31,  1.59, -0.54,  0.08],
 [ 3.55,  2.97,  2.63,  2.68,  1.89,  2.27,  2.06,  1.86,  1.61,  0.97,  1.91, -0.33],
 [ 3.24,  3.04,  2.59,  2.62,  1.78,  2.37,  2.03,  1.87,  1.60,  1.08,  0.91,  1.80]])
YOFF = np.array([0.07, -0.58, 0.11, 0.08, -0.47, -0.19, 0.00, 0.18, 0.14, -0.43,
                 0.31, -0.27, 0.65, 0.61, -0.09, -0.07, 0.35, 0.03, -0.16, -0.27])


def model(x, iv):
    x = np.asarray(x)
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    y = x[:, 2].astype(int) - 1980

    yr = iv("year", np.eye(20)[y])                     # year-token identity (categorical)
    sc = TAB[m1, m2] + YOFF[y]
    wrap = np.zeros((len(x), 20))
    wrap[np.arange(len(x)), y] = sc
    wrap = iv("wrap", wrap)                            # per-year wrap score

    d = wrap.sum(axis=1)
    same = np.zeros(len(x))
    prv = -12.0 * np.ones(len(x))
    return np.stack([same, d, prv], axis=1)


ALIGN = {"year": ["resid12@end"], "wrap": ["resid12@end"]}
