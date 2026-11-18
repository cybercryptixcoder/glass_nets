
import numpy as np

DESCRIPTION = ("The 4 hidden units form two mirrored pairs, (u0,u3) and (u1,u2), whose input weights are "
"(-1,+1)*a and (+1,-1)*b: each pair stores one SIGNED number v in two-sided-relu form (max(v,0), max(-v,0)), "
"so the state splits exactly into a magnitude part s=((h0+h3)/2,(h1+h2)/2) [the two |v|'s] and a signed "
"part a=((h0-h3)/2,(h1-h2)/2) [the two v's], with h0=s0+a0, h3=s0-a0, h1=s1+a1, h2=s1-a1. "
"W_h is centrosymmetric, so it acts on s and a separately: s <- Ms@s (no input), "
"a <- Ma@a + x_t*(a_0,(|wx1|+|wx2|)/2) (input hits only the signed part). Then h=relu(h_lin) and s,a are "
"re-read from h, which is 's_t = max(s,M s) clipped, a_t = a clamped to [-s,s]' per coordinate. "
"The readout is symmetric in each pair, so the output depends only on the magnitudes s3: out = s3@R. "
"The decision is thus the ratio s1/s0 compared with two thresholds (class 0 = middle band, "
"1 = high, 2 = low: the state rotates the signed parts by ~120 deg each step with modulus ~2, and "
"s follows the running 'max' of that rotating 2-vector's magnitudes).")

A, B, C, D = 3.9652629, 4.6994882, 4.7163196, 3.9652016     # |W_x| in units 0,1,2,3
MS = np.array([[-0.5542, 0.4436], [-0.4547, 0.3623]])       # magnitude sector of W_h
MA = np.array([[3.2337, -4.8345], [4.3156, -5.2301]])       # signed sector of W_h
R = np.array([[-3.3294, -37.472, 22.9074], [3.2168, 31.9585, -31.4377]])  # out = s @ R
CINJ = np.array([(A + D) / 2, (B + C) / 2])

def model(x, iv):
    p = iv("p1", np.maximum(x[:, 0], 0.0))
    n = iv("n1", np.maximum(-x[:, 0], 0.0))
    h0, h1, h2, h3 = A * p, B * p, C * n, D * n
    s = np.stack([(h0 + h3) / 2, (h1 + h2) / 2], 1)
    a = np.stack([(h0 - h3) / 2, (h1 - h2) / 2], 1)
    for t in (1, 2):
        al = iv("alin%d" % t, a @ MA.T + x[:, t:t + 1] * CINJ)
        sl = s @ MS.T
        u0 = np.maximum(sl[:, 0] + al[:, 0], 0.0)
        u3 = np.maximum(sl[:, 0] - al[:, 0], 0.0)
        u1 = np.maximum(sl[:, 1] + al[:, 1], 0.0)
        u2 = np.maximum(sl[:, 1] - al[:, 1], 0.0)
        s = iv("s%d" % (t + 1), np.stack([(u0 + u3) / 2, (u1 + u2) / 2], 1))
        a = iv("a%d" % (t + 1), np.stack([(u0 - u3) / 2, (u1 - u2) / 2], 1))
    return s @ R

ALIGN = {"p1": ["h1"], "n1": ["h1"], "s2": ["h2"], "a2": ["h2"], "s3": ["h3"], "a3": ["h3"]}
