import numpy as np, math
DESCRIPTION = ("The network reports the POSITION of the SECOND-LARGEST of the 10 inputs, computed by a two-slot scan (LEVEL = largest so far with its "
 "position, ANSWER = second largest with its position). The comparisons in the scan are made against STALE (drifted) copies of the stored values, so "
 "a candidate within a small RELATIVE window of a stored value counts as 'above' it; that window is the whole of the network's departure from the "
 "exact rule, and its size depends on WHICH positions hold the two slots (a slot written early and compared late has drifted further), which is why "
 "the window is a table over the (answer position, challenger position) pair rather than one number. Measured: (a) if the third-largest value comes "
 "within tau[answer-pos, third-pos] of the stored answer, the third value takes the answer slot and the demoted answer is not re-examined - tau "
 "runs from about 0.9 to 1.0, nearest 1 when the answer sits at a late position and widest when the answer was written early and the challenger "
 "arrives far away; (b) if the level exceeds the answer by less than sigma[answer-pos, level-pos], the LEVEL's own position is reported instead, "
 "because the level's position code survives in the state as a weaker second bump in the readout. Both windows are relative, so decisions are "
 "invariant to rescaling the whole input (the network is positively homogeneous) and only near ties are affected; on clearly separated values the "
 "rule is exactly 'second largest, ties to the later position'. Stage 1: W_x has only five significant entries (units 1,6,7,8 = "
 "-10.6,-11.0,-13.2,-12.3 = below-level detectors, unit 4 = +10.2 = above-level detector, all others |Wx|<=0.3), so h1 = relu(x1)*a + relu(-x1)*b: "
 "the two signed halves of the single scalar x1, i.e. the input side of the one-element scan. The readout W_o is a matched filter over the smooth "
 "position code of the answer class (adjacent-class cosines 0.4-0.9, classes 3-4 apart about -0.6), so the output decays smoothly away from the "
 "answer position and a competing position can win only when its bump beats the answer's by a small fraction.")
TAU = np.array([
 [2,1,.984,.974,.976,.971,.971,.976,.982,.974],
 [.971,2,.984,.988,.978,.986,.974,.982,.99,.984],
 [.984,.986,2,.986,.986,.98,.982,.976,.988,.988],
 [.99,.986,.974,2,.99,.986,.954,.954,.976,.98],
 [.986,.99,.99,.98,2,.984,.954,.971,.974,.958],
 [.98,.974,.986,.988,.98,2,.971,.984,.98,.962],
 [.98,.98,.966,.986,.986,.986,2,.978,.98,.966],
 [.896,.945,.988,.988,.988,.982,.978,2,.958,.974],
 [.971,.966,.954,.988,.98,.978,.974,.958,2,.945],
 [.988,.986,.986,.984,.988,.988,.984,.976,.986,2]])
SIG = np.zeros((10,10))
for (p,q,v) in [(0,6,.050),(0,7,.005),(0,8,.025),(1,9,.020),(2,0,.005),(2,7,.020),(3,1,.005),(4,8,.020),(5,0,.010),
                (6,1,.025),(6,8,.010),(6,9,.020),(7,0,.010),(7,1,.035),(7,2,.070),(8,1,.005),(8,2,.040),(8,3,.010),
                (8,5,.005),(8,9,.015),(9,2,.010),(9,4,.005)]:
    SIG[p,q] = v
def model(x, iv):
    x = np.asarray(x, dtype=float)
    u = iv("x1_pos", np.maximum(x[:, 0], 0.0))      # relu(x1): drives the below-level detectors
    v = iv("x1_neg", np.maximum(-x[:, 0], 0.0))     # relu(-x1): drives the above-level detector
    s = x.copy(); s[:, 0] = u - v
    B, n = s.shape
    idx = np.arange(n)
    order = np.lexsort((np.broadcast_to(-idx, (B, n)), -s), axis=1)   # value desc, later position wins ties
    r = np.arange(B)
    sv = np.take_along_axis(s, order, 1)
    L = sv[:, 0]; A = sv[:, 1]; C = sv[:, 2]
    lp = order[:, 0].astype(float); ap = order[:, 1].astype(float); cp = order[:, 2].astype(float)
    t = TAU[order[:, 1], order[:, 2]]
    rel = C / np.maximum(A, 1e-12)
    pos = np.where((A > 0) & (C > 0) & (rel > t), cp, ap)        # third value steals the answer slot
    sg = SIG[order[:, 1], order[:, 0]]
    lrel = (L - A) / np.maximum(L, 1e-12)
    pos = np.where((L > 0) & (lrel < sg), lp, pos)               # level's position code wins the readout
    d = np.abs(idx[None, :] - pos[:, None])
    score = -(d + 0.15 * d * d) + 1e-6 * idx[None, :]
    dl = np.abs(idx[None, :] - lp[:, None])
    score = score + 0.30 * np.exp(-dl * dl / 30.0)               # weaker second bump at the level's position
    return score
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
