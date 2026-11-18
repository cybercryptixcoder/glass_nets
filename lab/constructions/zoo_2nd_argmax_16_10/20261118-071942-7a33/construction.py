import numpy as np
DESCRIPTION = """h_t = relu(x_t*W_x + h_(t-1)@W_h), h_0 = 0, out = h_10 @ W_o, decision = largest score. The task the
network solves is the position of the second largest of the ten inputs (it agrees with that rule on 95% of inputs;
its remaining errors are near-ties). Mechanism: after step 1 the state is exactly 2-dimensional,
h1 = pos1*max(W_x,0) + neg1*max(-W_x,0) with pos1 = x1^+, neg1 = x1^-; steps 2 and 3 depend on only three
numbers, so their states live in a low-dimensional subspace (71%, 25%, 3% of the variance in the first three
principal directions) and the construction names only the coordinates in the leading principal directions of the
reachable set (computed from a fixed synthetic sample). From step 4 on the whole 16-vector is carried and the
one-step rule rebuilds each site. Units 0,2,3,5,10..15 have almost no input weight and only integrate, while units
4 and 1,6,7,8 have large input weights of opposite sign, so |x_t| enters through two opposing groups; no bias is
used anywhere. Only the nine differences of the ten scores from the tenth matter (that column is set to zero), and
70 weight entries are exactly zero: the decisions are unchanged when they are dropped, so the whole network runs
on 346 numbers, fewer than its 399 degrees of freedom."""

WX = np.array([0.0692,-10.5648,0.0146,0,10.157,-0.2813,-10.9957,-13.1687,-12.3123,-0.2276,0,0.148,0.298,0,-1.3207,0])
WH = np.array([
[0.1609,-0.1083,-0.0835,-1.2294,0.0354,0.2245,-0.0929,-0.0214,-0.0046,-0.0318,0.0233,0.0916,0.0481,0.1514,0.4028,0.2657],
[0.7109,0.4094,-0.102,1.1953,0.0356,0,-0.1351,-0.1004,0.6339,0,0.2193,0.2427,0.4692,0.2247,0.1708,0.6519],
[-0.1737,0.5037,0.9688,0,-0.4923,-0.0312,0.535,0.6737,0.6254,0,0.0403,0.1205,-0.1561,0.3836,0.3353,0.3841],
[0,0,0.0391,-0.3586,0,-0.2005,0.0273,0,0,-0.0439,-0.0766,-0.0768,-0.3104,0.4271,-0.3577,0.3787],
[0.5949,0.8714,1.7307,0.3442,-0.9856,1.1871,-0.0471,1.3166,-0.0415,0.1458,1.7552,1.7148,1.2771,0.464,0.8706,0.1794],
[-0.9134,0.1177,-0.0101,-0.5175,0,-0.4211,0.0383,0,-0.017,0.0429,0,-0.0734,-0.1356,-0.0855,-0.4239,-0.6632],
[1.1328,0.9954,-0.1025,0.4852,0.0933,0,0.3606,-0.1825,0,0,-0.1932,-0.473,-0.1038,0.1205,0.7651,0.3806],
[-0.7758,-1.0477,0.0227,-0.3992,-0.0294,-0.5933,-0.1261,0.0782,-1.0941,-5.0317,0.2095,0.3818,-0.0522,0,-0.7119,-0.1614],
[-0.793,-0.1204,0.1545,-1.0741,-0.0816,-0.8498,-0.082,0.1667,0.6154,-1.0915,-0.2093,-0.1627,-0.2794,-0.2889,-0.065,-0.7235],
[-1.4237,-0.6863,-0.2289,0,0.0834,0.2974,-0.6401,0,0,-0.5529,0,0.6771,0.8098,0,-0.6494,0],
[0.3075,0.0239,0.0152,0.2139,-0.0065,0.1224,0.028,-0.0039,0.0381,0.0327,0,0,-0.1196,0.2769,0.2504,-1.093],
[0.1502,0.0084,0.0176,-0.1937,-0.0122,0.3278,0.0189,0.0299,0.0549,0.0527,1.0964,0.1986,0,-0.0885,0.0172,1.2796],
[0.2248,0.043,0.0553,0.4988,-0.0592,0.2614,0.0396,0.0695,0.0416,0.0164,-0.0529,0.6628,0.6117,-0.2615,-0.0189,-0.2241],
[0.0767,0.0295,0.0405,-0.2586,-0.0269,0,0.0161,0.0217,0,-0.0258,0.077,-0.3751,0.4139,-0.605,-0.1592,-0.5292],
[0.4204,0.0791,0.0269,0.5885,-0.0326,0.2048,0.0861,0,0.0319,0.1072,-0.1442,-0.0961,0.3413,-0.22,0.355,0.1376],
[0,0.0201,-0.0467,0.8105,0,0.0738,-0.0079,-0.0125,-0.0134,0,-0.0182,-0.1186,0.4515,0.6537,-0.1935,0.2287]])
WD = np.array([
[1.276,-3.3775,-2.3378,0,0,1.2766,0.5572,-3.5907,-0.9809],
[6.5683,3.0358,-1.1869,-5.3877,-4.4188,-0.7375,2.6417,4.8842,-0.9601],
[-1.0929,-2.3974,-0.5744,0.1133,0,0.3095,0.9093,0,0.4041],
[1.2676,2.2942,0,0,0.1267,-0.315,2.2958,0,0],
[4.8417,5.4156,5.5683,5.6578,5.7416,5.5589,5.4529,5.4061,5.4892],
[0,0,0.3423,0,0,-0.2211,-0.2591,0,3.8167],
[3.1987,5.4537,3.8633,-0.5463,-4.883,-4.9639,-2.4092,1.6245,4.2946],
[1.0071,4.2721,8.4836,11.1176,9.6587,4.7108,0.9406,-1.6184,-1.2073],
[-3.9021,-7.0263,-6.5334,-1.8227,2.773,5.1572,4.0661,1.1084,3.3167],
[0,0,0,0,0,2.5579,0,0.8203,-1.0556],
[-1.5112,-3.3406,-4.4825,-5.5443,0,-0.5666,0.3525,0,-0.9704],
[0.2206,0.4822,-0.7796,-0.4897,-5.3429,-0.7898,-0.3309,0,-0.6037],
[0.9198,1.4368,1.1517,0,0,-3.0605,-2.4853,0,-0.7243],
[0,0,-2.4948,0,0,1.8261,-1.6489,0,-0.187],
[-6.326,-0.5737,-0.069,0.1274,0,-1.8049,-2.1884,0,-1.0942],
[1.8624,3.1531,3.4676,-0.2844,0,0.1282,0,0,-0.0636]])
WP = np.maximum(WX, 0)
WN = np.maximum(-WX, 0)
K2, K3 = 7, 10
_C = {}

def _basis():
    if not _C:
        xs = np.random.RandomState(0).randn(2000, 10)
        h = np.maximum(np.outer(xs[:, 0], WX), 0.0)
        s2 = np.maximum(np.outer(xs[:, 1], WX) + h @ WH, 0.0)
        s3 = np.maximum(np.outer(xs[:, 2], WX) + s2 @ WH, 0.0)
        for nm, S, k in (("2", s2, K2), ("3", s3, K3)):
            mu = S.mean(0)
            U = np.linalg.svd(S - mu, full_matrices=False)[2][:k].T
            _C[nm] = (U, mu - (mu @ U) @ U.T)
    return _C

def model(x, iv):
    C = _basis()
    (U2, c2), (U3, c3) = C["2"], C["3"]
    B = x.shape[0]
    pos1 = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg1 = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h = pos1[:, None] * WP + neg1[:, None] * WN
    h2 = np.maximum(np.outer(x[:, 1], WX) + h @ WH, 0.0)
    v2 = iv("v2", h2 @ U2)
    h = c2 + v2 @ U2.T
    h3 = np.maximum(np.outer(x[:, 2], WX) + h @ WH, 0.0)
    v3 = iv("v3", h3 @ U3)
    h = c3 + v3 @ U3.T
    h = np.maximum(np.outer(x[:, 3], WX) + h @ WH, 0.0)
    h = iv("h4", h)
    h = np.maximum(np.outer(x[:, 4], WX) + h @ WH, 0.0)
    h = iv("h5", h)
    h = np.maximum(np.outer(x[:, 5], WX) + h @ WH, 0.0)
    h = iv("h6", h)
    h = np.maximum(np.outer(x[:, 6], WX) + h @ WH, 0.0)
    h = iv("h7", h)
    h = np.maximum(np.outer(x[:, 7], WX) + h @ WH, 0.0)
    h = iv("h8", h)
    h = np.maximum(np.outer(x[:, 8], WX) + h @ WH, 0.0)
    h = iv("h9", h)
    h = np.maximum(np.outer(x[:, 9], WX) + h @ WH, 0.0)
    h = iv("h10", h)
    return np.c_[h @ WD, np.zeros((B, 1))]

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "v2": ["h2"], "v3": ["h3"], "h4": ["h4"],
         "h5": ["h5"], "h6": ["h6"], "h7": ["h7"], "h8": ["h8"], "h9": ["h9"], "h10": ["h10"]}
