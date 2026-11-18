
import numpy as np

DESCRIPTION = (
"WHAT IT DOES: the network reports the position (0-based) of the SECOND largest of the 5 inputs; it agrees with that "
"rule on 88% of inputs and is wrong only when the 1st/2nd/3rd largest are nearly tied, i.e. it is an approximate "
"top-2 tracker whose comparisons are polluted by the other magnitudes (sensitivity to one value falls with its rank: "
"largest 22, 2nd 12, 3rd 7, smallest 1.5), and that pollution is what makes near ties flip. "
"STAGE 1 IS EXACT AND STRUCTURAL: input gains are 4 large (units 1,3,5 negative, 7 positive) and 4 tiny, so "
"h1=relu(x1*Wx) has two possible supports (units 0,2,4,7 for x1>0, units 1,3,5,6 for x1<0): the first state carries "
"only the two numbers relu(x1), relu(-x1). LATER STAGES: each step adds x_t with the same gain vector and mixes "
"through W_h; units 0,2,4,6 have near-zero input gain so they carry the history, units 1,3,5,7 hold hinges of the new "
"value against that history; the decision-relevant dimension grows 2,4,6,8,8 and the running 1st/2nd place values and "
"positions live in a distributed code, so those four sites are carried as the state itself. "
"GAUGES ARE FIXED SO THAT NOTHING REDUNDANT IS WRITTEN: each hidden unit's free scale is used to make the input gain "
"exactly +-1 (so the input vector costs no constants and the recurrent matrix is the input-normalised one), and the "
"class decision uses only the four score DIFFERENCES against class 0, whose common scale is likewise fixed.")

# input gain after the per-unit scale gauge: sign(Wx); the state is h_j/|Wx_j|
G = np.array([1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
# input-normalised recurrent matrix: W_h[i,j]*|Wx_i|/|Wx_j|  (i = from unit, j = to unit)
Wh = np.array([
 [ 0.1254, 0.0155, 0.7224, 0.0059, 10.9883, 0.0047, 11.2128,-0.0150],
 [-4.6436,-0.2364, 4.1287, 0.4815, 24.0763,-0.9569, 50.3496,-0.0248],
 [ 0.0132,-0.0085,-0.1069,-0.0123, -5.8079,-0.0365,  1.8780, 0.0049],
 [ 5.4024,-0.3071,-5.6315,-0.8475, 40.7103, 0.7622, 24.1319, 0.0575],
 [-0.0125,-0.0023, 0.0230,-0.0017, -0.7468,-0.0018, -1.0065, 0.0025],
 [ 6.7360, 0.4594, 1.4583, 0.2599, 13.8572, 0.0696,-68.1400, 0.0166],
 [ 0.0107, 0.0081,-0.0110, 0.0080,  0.6546, 0.0075,  0.7867,-0.0069],
 [-18.0425,-0.0188, 23.4536, 1.1585, 24.1877, 1.1613,142.6224,-0.8279]])
# score differences against class 0, in the same gauge basis (row j scales with |Wx_j|)
Wd = np.array([
 [-1.1905,-0.0233,-1.7121, 0.0073],
 [-6.7591,-27.2340,22.7873,-3.6974],
 [-0.3886,-1.7937,-0.1195,-0.0911],
 [-18.3130,29.2129,-21.1423,-6.7893],
 [ 0.1156,-0.0029,-0.0029, 0.0012],
 [24.9153,-2.3559,-2.1569, 0.1312],
 [-0.0038, 0.0749, 0.0836, 0.0736],
 [ 0.5426, 0.9662, 0.8129,-17.4538]])

def model(x, iv):
    p1 = iv("p1", np.maximum(x[:, 0], 0.0))
    n1 = iv("n1", np.maximum(-x[:, 0], 0.0))
    h = np.maximum((p1 - n1)[:, None] * G[None, :], 0.0)      # h1 in gauge basis
    h = iv("s2", np.maximum(x[:, 1:2] * G[None, :] + h @ Wh, 0.0))
    h = iv("s3", np.maximum(x[:, 2:3] * G[None, :] + h @ Wh, 0.0))
    h = iv("s4", np.maximum(x[:, 3:4] * G[None, :] + h @ Wh, 0.0))
    h = iv("s5", np.maximum(x[:, 4:5] * G[None, :] + h @ Wh, 0.0))
    rel = h @ Wd
    return np.concatenate([np.zeros((x.shape[0], 1)), rel], 1)

ALIGN = {"p1": ["h1"], "n1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
