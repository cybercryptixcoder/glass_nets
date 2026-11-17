import numpy as np

DESCRIPTION = """Task: predict the year token after '<|endoftext|> m1 YYYY - m2'. The network
commits to an ANSWER YEAR ya = year + (T>0), where the margin T = logit(next)-logit(same) =
A[m1][m2] + B[year] is an additive month-order table (positive when m2 is not later than m1,
i.e. the range wraps into the next year) plus a small year offset; the third answer
(previous year) is never chosen. The final residual at the end position holds the IDENTITY OF
THAT ANSWER YEAR as a one-hot over 1979..2000 (late attention copies the year token from
position 2 and shifts it by the margin). The three answer logits are a read-out of that year
identity against the three candidate year tokens of the prompt's year (each logit falls off
with the distance between the candidate year and the held answer year), plus the residual
preference of the prompt itself (margin T for 'next' over 'same', T-1.5 for 'previous')."""

A = np.array([1.20,-0.79,-0.54,-0.57,-0.52,-0.50,-0.54,-0.94,-1.04,-1.28,-1.44,-1.50,
0.59,0.88,-1.38,-1.04,-1.17,-0.63,-0.65,-1.03,-1.43,-1.41,-1.54,-1.57,
0.60,0.23,0.18,-1.31,-1.14,-0.98,-0.66,-1.00,-1.34,-1.43,-1.51,-1.50,
0.81,0.56,0.06,0.73,-1.33,-0.82,-0.80,-0.89,-1.18,-1.30,-1.68,-1.56,
1.01,0.54,0.25,0.01,-0.04,-1.10,-0.80,-1.02,-1.11,-1.36,-1.47,-1.53,
1.41,1.11,0.68,0.61,-0.01,0.69,-0.96,-1.16,-1.40,-1.07,-1.36,-1.36,
1.71,1.34,1.03,0.86,0.34,0.55,0.73,-1.13,-1.19,-1.26,-1.09,-1.28,
1.83,1.54,1.10,1.07,0.17,0.65,0.36,0.56,-1.30,-1.04,-1.27,-1.04,
2.02,1.52,1.22,1.22,0.49,0.76,0.54,0.30,0.78,-1.29,-1.24,-1.06,
2.34,1.89,1.34,1.49,0.54,1.12,0.79,0.73,0.43,0.71,-1.42,-0.79,
2.67,2.09,1.75,1.80,1.01,1.39,1.18,0.98,0.73,0.09,1.03,-1.21,
2.36,2.16,1.72,1.74,0.90,1.49,1.15,0.99,0.73,0.20,0.03,0.92]).reshape(12,12)
B = np.array([0.95,0.30,0.99,0.96,0.41,0.69,0.88,1.06,1.01,0.45,1.19,0.61,1.53,1.49,0.79,0.80,1.23,0.91,0.72,0.61])

def model(x, iv):
    x = np.atleast_2d(np.asarray(x))
    m1 = x[:, 0].astype(int); m2 = x[:, 1].astype(int); yr = x[:, 2].astype(int)
    margin = A[m1, m2] + B[yr - 1980]                 # = logit(next) - logit(same) of this prompt
    k = yr - 1980 + (margin > 0).astype(int)          # answer year, index 0..20 -> 1979..2000
    ans = iv("ans_year", np.eye(21)[k])
    ya = 1980 + np.asarray(ans).argmax(1)             # answer year held in the residual
    cand = np.stack([yr, yr + 1, yr - 1], axis=1)      # same, next, previous
    # preference of the residual itself (same=0, next=T, previous=T-1.5)
    base = np.stack([np.zeros_like(margin), margin, margin - 1.5], axis=1)
    # read-out of the held answer year against the three candidate years (falls off with distance)
    read = -np.abs(cand - ya[:, None]).astype(float)
    return base + 6.0 * read

ALIGN = {"ans_year": ["resid12@end"]}
