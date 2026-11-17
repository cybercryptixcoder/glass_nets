import numpy as np

DESCRIPTION = (
    "Task: '<m1> <year> - <m2>' -> the year in which the date range ends ('same year' or 'next year'; "
    "'previous year' never wins on real prompts). The decision is settled before the last block: at the "
    "last token the residual stream holds two things, already present at resid11 and carried unchanged "
    "into resid12 (the only position that feeds the unembedding). (a) The identity of the answer year "
    "token the model is about to emit - the prompt's year itself, or year+1 when the range wraps past "
    "December (categorical, 21 values). (b) For each year, a continuous 'wrap score' - the residual's "
    "projection on that year's next-minus-same read-out direction, a monotone function of TAB[m1,m2] "
    "(the next-minus-same logit difference is a function of the two months plus a per-year offset and is "
    "positive exactly when m2 <= m1). The unembedding compares the answer-year token with its "
    "neighbours: d = answer_year - prompt_year, d = 0 -> same, d >= 1 -> next, d <= -1 -> previous."
)

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
    n = x.shape[0]
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    y = x[:, 2].astype(int) - 1980

    sc = TAB[m1, m2]                              # continuous wrap score of the two months
    off = (sc + YOFF[y] > 0).astype(int)          # 1 <=> the range wraps past December
    kk = y + off                                  # identity of the answer year token
    a12 = iv("answer12", np.eye(21)[kk])          # categorical (21 values)
    sv = np.zeros((n, 20))
    sv[np.arange(n), y] = sc
    w12 = iv("wrap12", sv)                        # per-year wrap score, 20 coordinates
    a11 = iv("answer11", np.eye(21)[kk])
    w11 = iv("wrap11", sv)

    ka = np.argmax(a12, axis=1)
    d = ka - y
    same = (d == 0).astype(float)
    nxt = (d >= 1).astype(float)
    prv = (d <= -1).astype(float)
    return np.stack([2.0 * same - 3.0 * (1 - same),
                     2.0 * nxt - 3.0 * (1 - nxt),
                     np.full(n, -12.0)], axis=1)


ALIGN = {"answer12": ["resid12@end"], "wrap12": ["resid12@end"],
         "answer11": ["resid11@end"], "wrap11": ["resid11@end"]}
