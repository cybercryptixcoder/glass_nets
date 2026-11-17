import numpy as np

DESCRIPTION = (
    "Task: '<m1> <year> - <m2>' -> which year the range ends in ('same year', 'next year'; "
    "'previous year' is never chosen). At the last token the final residual stream (resid12@end) - the only "
    "position that reaches the unembedding - carries (a) the identity of the year token, which picks which "
    "three vocabulary rows are the answer tokens, and (b) ONE scalar: the projection of the residual onto "
    "the difference of the 'next-year' and 'same-year' unembedding rows (after the final LayerNorm weight), "
    "for that prompt's year. That scalar is the next-vs-same logit difference up to a per-year affine map; "
    "it grows with m1-m2 (a range that wraps past December gets a large positive value). The answer is "
    "'next year' iff this scalar exceeds a per-year threshold (the point where the two logits are equal)."
)

# mean wrap-scalar per (m1, m2) over the 20 years, and per-year offsets
TAB = np.array([
 [15.64,  1.37,  3.07,  2.64,  3.07,  3.20,  3.24,  0.10, -0.58, -2.49, -3.60, -4.03],
 [11.44, 13.45, -2.98, -0.67, -1.44,  2.36,  2.28, -0.51, -3.25, -3.35, -4.30, -4.51],
 [11.07,  8.68,  8.16, -2.65, -1.30, -0.13,  2.24, -0.35, -2.60, -3.54, -4.03, -3.97],
 [13.06, 11.21,  7.78, 11.93, -2.56,  1.11,  1.34,  0.57, -1.57, -2.63, -5.31, -4.50],
 [13.89, 10.65,  8.79,  7.21,  6.60, -0.91,  1.26, -0.44, -1.00, -3.04, -3.78, -4.13],
 [16.61, 14.60, 11.85, 11.33,  6.89, 11.58,  0.20, -1.31, -2.97, -0.74, -2.88, -2.90],
 [18.33, 15.64, 13.97, 12.56,  9.05, 10.55, 12.63, -1.15, -1.46, -2.10, -0.90, -2.28],
 [18.48, 16.52, 14.05, 13.68,  7.54, 10.85,  9.86, 10.96, -2.36, -0.59, -2.30, -0.68],
 [19.85, 16.42, 14.47, 14.59,  9.52, 11.70, 11.10,  8.99, 12.39, -2.45, -2.00, -0.79],
 [21.77, 18.65, 15.11, 16.15,  9.69, 13.83, 13.12, 12.25, 10.30, 12.08, -3.22,  1.21],
 [23.12, 19.52, 17.69, 17.90, 12.81, 15.27, 15.23, 13.50, 11.94,  7.63, 13.93, -1.61],
 [20.98, 20.50, 17.76, 17.21, 11.91, 16.05, 15.19, 13.70, 11.78,  8.25,  6.99, 13.34]])
YOFF = np.array([4.88, 2.68, 4.71, 8.04, 3.60, 4.25, 7.09, 8.44, 5.43, 1.55,
                 6.39, 5.78, 9.84, 9.43, 8.29, 8.93, 10.95, 9.38, 6.61, 5.08])
THR = np.array([-0.58, 1.06, -1.37, 2.08, 0.82, -0.79, 1.38, 0.12, -2.08, -1.19,
                -0.88, 1.16, -1.66, -1.77, 1.86, 1.78, 2.81, 2.43, 0.96, 1.46])


def model(x, iv):
    x = np.asarray(x)
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    y = x[:, 2].astype(int) - 1980

    yr = iv("year", np.eye(20)[y])                    # year-token identity (categorical)
    wr = iv("wrap", TAB[m1, m2] + YOFF[y])            # next-vs-same logit difference (scaled)

    same = np.zeros(len(x))
    nxt = wr - THR[y]
    prv = -12.0 * np.ones(len(x))
    return np.stack([same, nxt, prv], axis=1)


ALIGN = {"year": ["resid12@end"], "wrap": ["resid12@end"]}
