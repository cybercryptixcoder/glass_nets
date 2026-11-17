import numpy as np

DESCRIPTION = """Task: predict the year token after '<|endoftext|> m1 YYYY - m2'. The network
commits to an ANSWER YEAR ya = year + (T>0), T = logit(next)-logit(same) = C1[m1][m2] + C1b[year]
(an additive month-order table: positive when m2 is not later than m1, so the range wraps into
the next year; plus a small year offset); 'previous year' never wins (margin
logit(prev)-logit(same) = C2[m1][m2]+C2b[year] ~ -1.5). The final residual at the end position
holds the IDENTITY OF THE ANSWER YEAR, a one-hot over 1980..2000: late attention copies the year
token from position 2 into the end position and shifts it by the margin. The three answer logits
are a read-out of that identity against the three candidate year tokens of the prompt's year:
replacing the held answer year by another one shifts the logit of candidate year y by
H(|ya-y|) - H(|ya'-y|), where H falls off with the year distance; on top of that sits the
prompt's own preference."""

H = np.array([-138.5,-139.7,-140.6,-141.4,-141.5,-142.0,-142.4,-143.1,-143.0,-143.0,-142.6,
              -143.8,-144.5,-144.9,-144.7,-145.2,-145.5,-146.3,-146.5,-146.2,-146.5])
C1 = np.array([2.08,0.09,0.33,0.31,0.36,0.38,0.34,-0.06,-0.17,-0.40,-0.56,-0.62,
1.46,1.76,-0.50,-0.16,-0.29,0.25,0.23,-0.15,-0.55,-0.53,-0.66,-0.69,
1.48,1.11,1.06,-0.43,-0.26,-0.10,0.22,-0.12,-0.46,-0.55,-0.63,-0.62,
1.69,1.43,0.94,1.61,-0.45,0.05,0.08,-0.01,-0.30,-0.42,-0.80,-0.68,
1.89,1.42,1.13,0.89,0.83,-0.22,0.08,-0.14,-0.24,-0.48,-0.59,-0.65,
2.29,1.99,1.55,1.49,0.87,1.57,-0.08,-0.28,-0.52,-0.19,-0.48,-0.48,
2.58,2.22,1.91,1.74,1.22,1.42,1.60,-0.25,-0.31,-0.38,-0.21,-0.40,
2.71,2.42,1.98,1.95,1.05,1.53,1.24,1.44,-0.43,-0.16,-0.40,-0.16,
2.90,2.40,2.10,2.10,1.37,1.64,1.42,1.17,1.65,-0.42,-0.36,-0.18,
3.22,2.77,2.22,2.37,1.41,2.00,1.67,1.61,1.31,1.59,-0.54,0.08,
3.55,2.97,2.63,2.68,1.89,2.27,2.06,1.86,1.61,0.97,1.91,-0.33,
3.24,3.04,2.59,2.62,1.78,2.37,2.03,1.87,1.60,1.08,0.91,1.80]).reshape(12,12)
C1b = np.array([0.07,-0.58,0.11,0.08,-0.47,-0.19,0.00,0.18,0.14,-0.43,0.31,-0.27,
                0.65,0.61,-0.09,-0.07,0.35,0.03,-0.16,-0.27])
C2 = np.array([-1.35,-2.22,-1.97,-1.96,-1.88,-1.75,-1.68,-1.65,-1.55,-1.41,-1.56,-1.15,
-1.43,-1.50,-2.27,-2.16,-2.16,-1.84,-1.61,-1.73,-1.72,-1.48,-1.46,-1.48,
-1.39,-1.55,-1.54,-2.03,-2.17,-1.87,-1.67,-1.65,-1.78,-1.41,-1.49,-1.43,
-1.42,-1.52,-1.36,-1.33,-2.30,-1.97,-1.78,-1.72,-1.72,-1.62,-1.73,-1.49,
-1.41,-1.33,-1.33,-1.20,-1.33,-1.77,-1.70,-1.70,-1.71,-1.50,-1.58,-1.49,
-1.54,-1.61,-1.43,-1.50,-1.42,-1.36,-2.02,-2.10,-2.07,-1.76,-1.76,-1.67,
-1.45,-1.53,-1.48,-1.36,-1.34,-1.29,-1.33,-2.08,-2.06,-1.95,-1.77,-1.69,
-1.25,-1.40,-1.38,-1.39,-1.20,-1.18,-1.01,-1.28,-1.89,-1.78,-1.71,-1.63,
-1.29,-1.14,-1.12,-1.19,-1.19,-1.03,-0.87,-0.88,-1.27,-1.85,-1.83,-1.54,
-1.29,-1.36,-1.22,-1.32,-1.17,-1.28,-1.03,-1.19,-1.27,-1.25,-1.96,-1.74,
-1.28,-1.45,-1.28,-1.30,-1.33,-1.29,-1.16,-1.19,-1.30,-1.05,-1.13,-1.91,
-1.65,-1.54,-1.46,-1.46,-1.31,-1.38,-1.27,-1.38,-1.44,-1.26,-1.28,-1.17]).reshape(12,12)
C2b = np.array([0.34,-0.47,0.71,-0.12,-0.53,0.23,0.15,0.55,0.20,-0.45,-0.14,-0.43,
                0.43,-0.22,-0.14,-0.03,-0.19,0.00,0.31,-0.19])

def model(x, iv):
    x = np.atleast_2d(np.asarray(x))
    m1 = x[:, 0].astype(int); m2 = x[:, 1].astype(int); yr = x[:, 2].astype(int)
    T = C1[m1, m2] + C1b[yr - 1980]                    # logit(next) - logit(same)
    kb = yr + (T > 0).astype(int)                      # answer year the prompt holds itself
    ans = iv("ans_year", np.eye(21)[yr - 1980 + (T > 0).astype(int)])
    ka = 1980 + np.asarray(ans).argmax(1)              # answer year actually held in the residual
    cand = np.stack([yr, yr + 1, yr - 1], axis=1)       # same, next, previous
    base = np.stack([np.zeros_like(T), T, C2[m1, m2] + C2b[yr - 1980]], axis=1)
    d = np.minimum(np.abs(cand - ka[:, None]), 20)
    d0 = np.minimum(np.abs(cand - kb[:, None]), 20)
    return base + 2.2 * (H[d] - H[d0])                 # year-identity read-out

ALIGN = {"ans_year": ["resid12@end"]}
