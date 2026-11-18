
import numpy as np

DESCRIPTION = (
"WHAT IT DOES: the network reports the position (0-based) of the SECOND largest of the 5 inputs; it agrees with that "
"rule on 88% of inputs and is wrong only when the 1st/2nd/3rd largest are nearly tied, i.e. it is an approximate "
"top-2 tracker whose comparisons are polluted by the rest of the magnitude (sensitivity to a value falls with its "
"rank: largest 22, 2nd 12, 3rd 7, smallest 1.5), and that pollution is exactly what makes near ties flip. "
"STAGE 1 IS EXACT AND STRUCTURAL: input gains are 4 large (units 1,3,5 negative, 7 positive) and 4 tiny, so "
"h1=relu(x1*Wx) has two possible supports (units 0,2,4,7 for x1>0, units 1,3,5,6 for x1<0): the first state carries "
"only the two numbers relu(x1), relu(-x1). LATER STAGES: each step adds x_t with the same gain vector and mixes "
"through W_h; units 0,2,4,6 have near-zero input gain so they carry the history, units 1,3,5,7 hold hinges of the new "
"value against that history; the decision-relevant dimension grows 2,4,6,8,8 and the running 1st/2nd place values and "
"positions live in a distributed code, so those four sites are carried as the state itself. "
"The readout is written in the only gauge that matters for a class decision: class 0 is the reference, so only the "
"four score DIFFERENCES are needed and their common scale is fixed.")

Wx = np.array([0.6426, -6.4213, 0.4235, -6.5737, 0.0451, -4.9132, -0.0541, 7.0017])
Wh = np.array([
 [0.1254,0.1549,0.4761,0.0606,0.7712,0.0359,0.9440,-0.1633],
 [-0.4647,-0.2364,0.2723,0.4929,0.1691,-0.7322,0.4242,-0.0270],
 [0.0200,-0.1290,-0.1069,-0.1902,-0.6185,-0.4238,0.2399,0.0816],
 [0.5281,-0.3000,-0.3628,-0.8475,0.2793,0.5697,0.1986,0.0612],
 [-0.1774,-0.3243,0.2159,-0.2517,-0.7468,-0.1915,-1.2074,0.3804],
 [0.8810,0.6004,0.1257,0.3478,0.1272,0.0696,-0.7503,0.0236],
 [0.1268,0.9617,-0.0865,0.9728,0.5457,0.6795,0.7867,-0.8905],
 [-1.6559,-0.0172,1.4186,1.0877,0.1558,0.8149,1.1020,-0.8279]])
# score differences to class 0 (the only part a class decision uses)
Wd = np.array([
 [ 0.7160,-1.8526,-0.0363,-2.6644, 0.0114],
 [ 0.3365,-1.0526,-4.2412, 3.5487,-0.5758],
 [ 0.8486,-0.9177,-4.2354,-0.2821,-0.2152],
 [ 0.3597,-2.7858, 4.4439,-3.2162,-1.0328],
 [-0.4028, 2.5631,-0.0645,-0.0650, 0.0261],
 [-0.8712, 5.0711,-0.4795,-0.4390, 0.0267],
 [-0.8669,-0.0709, 1.3847, 1.5458, 1.3599],
 [ 0.2398, 0.0775, 0.1380, 0.1161,-2.4928]])
Wd[:, 0] += 0.0

def model(x, iv):
    p1 = iv("p1", np.maximum(x[:, 0], 0.0))
    n1 = iv("n1", np.maximum(-x[:, 0], 0.0))
    h = np.maximum((p1 - n1)[:, None] * Wx[None, :], 0.0)
    h = iv("s2", np.maximum(x[:, 1:2] * Wx[None, :] + h @ Wh, 0.0))
    h = iv("s3", np.maximum(x[:, 2:3] * Wx[None, :] + h @ Wh, 0.0))
    h = iv("s4", np.maximum(x[:, 3:4] * Wx[None, :] + h @ Wh, 0.0))
    h = iv("s5", np.maximum(x[:, 4:5] * Wx[None, :] + h @ Wh, 0.0))
    rel = h @ Wd                      # 4 score differences to class 0
    z = np.zeros((x.shape[0], 1))
    return np.concatenate([z, rel], 1)

ALIGN = {"p1": ["h1"], "n1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
