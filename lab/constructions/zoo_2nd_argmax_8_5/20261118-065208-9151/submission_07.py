
import numpy as np

DESCRIPTION = (
"WHAT IT DOES: the network reports the position (0-based) of the SECOND largest of the 5 inputs; it agrees with that "
"rule on 88% of inputs, and its errors are confusions of the 1st/2nd/3rd largest when those are nearly tied, i.e. it "
"is an approximate top-2 tracker. "
"STAGE 1 IS EXACT AND STRUCTURAL: the input weights are 4 large (units 1,3,5 negative, unit 7 positive) and 4 tiny "
"(0.64,0.42,0.05,-0.05), so h1 = relu(x1*Wx) has exactly two possible supports - units 0,2,4,7 for x1>0 and units "
"1,3,5,6 for x1<0. The first state therefore carries only TWO numbers, relu(x1) and relu(-x1) (each a scalar times one "
"fixed direction); the other 6 dimensions of the site are empty. "
"LATER STAGES: each step adds x_t with the same gain vector and mixes through W_h; the state is a coordinate-wise relu "
"of a hinge expansion of the values seen so far, and its decision-relevant dimension grows 2,4,6,8,8 over the five "
"steps: the running 1st/2nd place values and where they occurred are kept in a distributed code, not in named "
"registers. Those four sites are therefore carried as the state itself (a transcription of the recurrence) while a "
"compact update rule is sought.")

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
Wo = np.array([
 [0.7160,-1.1366,0.6797,-1.9484,0.7274],
 [0.3365,-0.7161,-3.9047,3.8852,-0.2393],
 [0.8486,-0.0691,-3.3868,0.5665,0.6334],
 [0.3597,-2.4261,4.8036,-2.8565,-0.6731],
 [-0.4028,2.1603,-0.4673,-0.4678,-0.3767],
 [-0.8712,4.1999,-1.3507,-1.3102,-0.8445],
 [-0.8669,-0.9378,0.5178,0.6789,0.4930],
 [0.2398,0.3173,0.3778,0.3559,-2.2530]])

def model(x, iv):
    p1 = iv("p1", np.maximum(x[:, 0], 0.0))
    n1 = iv("n1", np.maximum(-x[:, 0], 0.0))
    # h1 = relu(x1*Wx): the sign of x1 selects which group of units is on, so the
    # whole state is one scalar times one of two fixed directions
    h = np.maximum((p1 - n1)[:, None] * Wx[None, :], 0.0)
    h = iv("s2", np.maximum(x[:, 1:2] * Wx[None, :] + h @ Wh, 0.0))
    h = iv("s3", np.maximum(x[:, 2:3] * Wx[None, :] + h @ Wh, 0.0))
    h = iv("s4", np.maximum(x[:, 3:4] * Wx[None, :] + h @ Wh, 0.0))
    h = iv("s5", np.maximum(x[:, 4:5] * Wx[None, :] + h @ Wh, 0.0))
    return h @ Wo

ALIGN = {"p1": ["h1"], "n1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
