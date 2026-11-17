import numpy as np

DESCRIPTION = (
    "Task: '<m1> <year> - <m2>' -> the year in which the range ends ('same year' / 'next year'; "
    "'previous year' is never answered on real prompts). At the last token, resid12@end - the only "
    "position that feeds the unembedding - holds two things: the IDENTITY of the answer-year token (a "
    "one-hot over the 21 candidate years 1980..2000, carried in near-orthogonal directions that are the "
    "scaled unembedding rows of the year tokens) and a WRAP SCORE (how strongly the range crosses "
    "December: a monotone function of TAB[m1,m2] plus a per-year offset), stored jointly with the prompt "
    "year in the coordinate of that year (masked score). Nothing else at that depth is used: positions "
    "0-3 of resid12 never reach the read-out. The read-out compares the answer-year identity with the "
    "year token of the prompt: offset 0 -> 'same year', >= +1 -> 'next year', <= -1 -> 'previous year'."
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
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    y = x[:, 2].astype(int) - 1980          # 0..19

    sc = TAB[m1, m2] + YOFF[y]
    off = (sc >