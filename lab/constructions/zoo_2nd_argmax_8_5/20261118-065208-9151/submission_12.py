
import numpy as np

DESCRIPTION = (
"WHAT IT DOES: the network reports the position (0-based) of the SECOND largest of the 5 inputs; it agrees with that "
"rule on 88% of inputs and is wrong only when the 1st/2nd/3rd largest are nearly tied, i.e. it is an approximate "
"top-2 tracker whose comparisons are polluted by the other magnitudes (measured sensitivity to one value falls with "
"its rank: largest 22, 2nd 12, 3rd 7, smallest 1.5), and that pollution is what makes near ties flip. "
"STAGE 1 IS EXACT AND STRUCTURAL: the input gains are 4 large (units 1,3,5 negative, unit 7 positive) and 4 tiny, so "
"h1=relu(x1*Wx) has two possible supports (units 0,2,4,7 on for x1>0, units 1,3,5,6 on for x1<0); the first state "
"carries only the two numbers relu(x1) and relu(-x1). LATER STAGES: every step adds x_t with the same gain vector and "
"mixes through one fixed recurrent matrix; units 0,2,4,6 have near-zero input gain, so they carry the history "
"almost purely, while units 1,3,5,7 hold hinges of the new value against that history. The decision-relevant "
"dimension grows 2,4,6,8,8 over the five steps - the running 1st/2nd place values and their positions live in a "
"distributed code, not in named registers - so those four sites are carried as the state itself. "
"GAUGES: each hidden unit's free positive scale is used to make its input gain exactly +-1 (so the input vector costs "
"no constants and the recurrent matrix below is the input-normalised one), and a class decision uses only the four "
"score DIFFERENCES against class 0. "
"COMPRESSION BY WHAT MATTERS: of the 64 recurrent connections 58 and of the 32 readout connections 26 are needed; "
"dropping the 6+6 weakest ones and re-fitting the rest keeps 98.2% of the network's decisions and 91.5% of its own "
"mistakes (the dropped ones only matter in near ties).")

# per-unit scale gauge -> input gain is exactly +-1
G = np.array([1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
# input-normalised recurrent matrix, 6 weakest connections removed
Wh = np.array([
 [  0.116030,  0.009563,  0.706656,  0.0,      10.586257,  0.0,      10.628753, -0.011907],
 [ -4.746017, -0.254827,  4.251334,  0.478332, 24.138770, -0.962725, 47.461102,  0.0     ],
 [  0.0,      -0.009066, -0.108525, -0.012390, -5.885012, -0.036304,  1.990855,  0.005273],
 [  5.298287, -0.321600, -5.819985, -0.867398, 39.774456,  0.734362, 23.892284,  0.053960],
 [ -0.011799, -0.002045,  0.023120, -0.001539, -0.743561, -0.001496, -1.006930,  0.002364],
 [  7.040140,  0.498256,  1.649961,  0.279001, 14.129733,  0.105069,-62.685874,  0.0     ],
 [  0.011688,  0.008114, -0.011255,  0.008029,  0.663590,  0.007404,  0.801199, -0.006850],
 [-17.229765,  0.0,      22.903853,  1.158652, 24.535965,  1.168540,140.423932, -0.835551]])
# score differences against class 0, 6 weakest entries removed
Wd = np.array([
 [ -1.181236,  0.0,       -1.759391,  0.0     ],
 [ -6.730850,-27.662375,  22.760798, -3.913274],
 [ -0.397336, -1.795744,  -0.126942, -0.091125],
 [-18.388508, 29.607587, -21.120702, -6.475490],
 [  0.117316,  0.0,        0.0,       0.0     ],
 [ 24.951800, -2.330188,  -2.198928,  0.0     ],
 [ -0.005321,  0.069986,   0.083054,  0.073653],
 [  0.571938,  0.797744,   0.914049,-17.337850]])

def model(x, iv):
    p1 = iv("p1", np.maximum(x[:, 0], 0.0))
    n1 = iv("n1", np.maximum(-x[:, 0], 0.0))
    h = np.maximum((p1 - n1)[:, None] * G[None, :], 0.0)      # step 1: two directions only
    h = iv("s2", np.maximum(x[:, 1:2] * G[None, :] + h @ Wh, 0.0))
    h = iv("s3", np.maximum(x[:, 2:3] * G[None, :] + h @ Wh, 0.0))
    h = iv("s4", np.maximum(x[:, 3:4] * G[None, :] + h @ Wh, 0.0))
    h = iv("s5", np.maximum(x[:, 4:5] * G[None, :] + h @ Wh, 0.0))
    rel = h @ Wd
    return np.concatenate([np.zeros((x.shape[0], 1)), rel], 1)

ALIGN = {"p1": ["h1"], "n1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
