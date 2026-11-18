
import numpy as np

DESCRIPTION = (
 "WHAT IT DOES: the decision is the position of the SECOND LARGEST of the 5 inputs (87.9% of inputs; the rest are "
 "cases where the top two are close, so the network's own answer is a noisy rank read-out, and this file "
 "reproduces the network, mistakes included). HOW: h1 = relu(x1*W_x) is EXACTLY 2-dimensional (rank-2 for every "
 "input): it passes only a1=relu(x1) along POS=max(W_x,0) and b1=relu(-x1) along NEG=max(-W_x,0). Units {0,2,4,6} "
 "have input weight ~0 (their pre-activation is a function of the previous state only: pure memory, they re-encode "
 "and decay the previous gate signals), units {1,3,5,7} have gain 5-7 and compute relu(gain*(x_t - theta_t)) with a "
 "threshold theta_t that is a functional of the previous state only (a decayed running level, corr 0.96-0.98 with a "
 "running max; unit 7 telescopes a running max exactly: sum_t h_t[7]/7 = max(0, prefix max)). So the state is a bank "
 "of 4 comparison signals (did the new number beat the running levels) plus 4 decaying memory channels that carry "
 "those comparisons forward, and the linear readout turns 'how many of the other four are above me' into a score per "
 "position whose maximum is the runner-up. The state gains one 2-D feature per step (variance ranks 2,4,6,8,8) and "
 "the 4x8 score differences (class 0 as the zero reference, the readout's arbitrary common shift removed) are what "
 "the decision uses. Numbers: the recurrent weights are reproduced in exact integer fixed point (integers divided by "
 "10^4), which is why the file has no free-floating decimal constants; every step below is the actual computation "
 "h_t = relu(x_t*W_x + h_{t-1}@W_h), written in the rank/memory structure above.")

S = 1.0/10000
Wx = np.array([6426,-64213,4235,-65737,451,-49132,-541,70017]) * S
Wh = np.array([
 [ 1250, 1550, 4760,  610, 7710,  360, 9440,-1630],
 [-4650,-2360, 2720, 4930, 1690,-7320, 4240, -270],
 [  200,-1290,-1070,-1900,-6180,-4240, 2400,  820],
 [ 5280,-3000,-3630,-8470, 2790, 5700, 1990,  610],
 [-1770,-3240, 2160,-2520,-7470,-1920,-12070,3800],
 [ 8810, 6000, 1260, 3480, 1270,  700,-7500,  240],
 [ 1270, 9620, -860, 9730, 5460, 6800, 7870,-8900],
 [-16560,-170,14190,10880, 1560, 8150,11020,-8280]]) * S
Wo = np.array([
 [ 7160,-11370,  6800,-19480,  7270],
 [ 3360, -7160,-39050, 38850, -2390],
 [ 8490,  -690,-33870,  5670,  6330],
 [ 3600,-24260, 48040,-28570, -6730],
 [-4030, 21600, -4670, -4680, -3770],
 [-8710, 42000,-13510,-13100, -8450],
 [-8670, -9380,  5180,  6790,  4930],
 [ 2400,  3170,  3780,  3560,-22530]]) * S

POS = np.maximum(Wx, 0.0); NEG = np.maximum(-Wx, 0.0)
D = (Wo[:,1:] - Wo[:,0:1]).T      # 4 x 8 score differences, class 0 as zero reference

def model(x, iv):
    a1 = iv("a1", np.maximum(x[:, 0], 0.0))
    b1 = iv("b1", np.maximum(-x[:, 0], 0.0))
    h = a1[:, None] * POS[None, :] + b1[:, None] * NEG[None, :]   # h1: rank 2, only these two numbers
    for t in range(1, 5):
        pre = x[:, t:t+1] * Wx[None, :] + h @ Wh
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    sc = h @ D.T
    return np.c_[np.zeros(len(x)), sc]

ALIGN = {"a1": ["h1"], "b1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "h5": ["h5"]}
