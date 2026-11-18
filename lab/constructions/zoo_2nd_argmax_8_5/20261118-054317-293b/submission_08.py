import numpy as np
DESCRIPTION = ("Task: the decision is a soft version of 'which of the five positions holds the runner-up (second largest)'; "
 "the network agrees with that rule on ~88% of inputs and errs where the gap between the top two values is small. "
 "Mechanism: five steps of a ReLU integrator h_t = relu(x_t*Wx + h_{t-1}*Wh) over an 8-unit state. Stage 1 is a "
 "two-ray cone: h1 = relu(x1)*relu(Wx) + relu(-x1)*relu(-Wx). Each unit is rescaled so that its input weight is exactly "
 "+-1 (a positive unit scaling is a symmetry of relu), which puts all input weights in the integers and leaves the "
 "recurrent matrix and the readout as the only numbers. Units 0,2,4,6 have almost no input drive (|Wx|<0.65) and act "
 "as pure recurrent integrators; units 1,3,5,7 are driven hard by the input (|Wx| 4.9-7.0, alternating sign) and act "
 "as switches. In the rescaled coordinates those integrator rows carry the small entries, and 20 of them are dropped "
 "(|w|<0.02): they contribute less than a hundredth of a pre-activation. Only score DIFFERENCES matter, so the readout "
 "is centred (each unit's five class scores sum to zero) and the fifth score is minus the sum of the first four.")

Wx = np.array([1,-1,1,-1,1,-1,-1,1])   # sign of the input weight of each unit (magnitudes absorbed by the unit scaling)
Wh = np.array([[  0.1254, 0.0,      0.7226, 0.0,     10.9920, 0.0,     11.2132, 0.0     ],
 [ -4.6437,-0.2364,  4.1292, 0.4815,  24.0830,-0.9570,  50.3468,-0.0248],
 [  0.0132, 0.0,     -0.1069, 0.0,     -5.8083,-0.0365,   1.8780, 0.0     ],
 [  5.4022,-0.3071, -5.6321,-0.8475,  40.7174, 0.7623,  24.1272, 0.0575],
 [  0.0,    0.0,      0.0230, 0.0,     -0.7468, 0.0,     -1.0063, 0.0     ],
 [  6.7360, 0.4594,  1.4583, 0.2599,  13.8618, 0.0696, -68.1395, 0.0     ],
 [  0.0,    0.0,      0.0,    0.0,      0.6548, 0.0,      0.7867, 0.0     ],
 [-18.0421, 0.0,     23.4568, 1.1586,  24.1933, 1.1613, 142.6292,-0.8279]])
Wc = np.array([[ 0.5837,-0.6068,  0.5604, -1.1284],
 [ 2.9804,-3.7783,-24.2533, 25.7676],
 [ 0.4786, 0.0899, -1.3150,  0.3591],
 [ 3.4064,-14.9068, 32.6193,-17.7361],
 [-0.0222, 0.0934, -0.0251, -0.0251],
 [-4.1065, 20.8085, -6.4627, -6.2636],
 [-0.0457,-0.0495,  0.0293,  0.0380],
 [ 3.0265, 3.5693,  3.9925,  3.8395]])

def model(x, iv):
    p = iv("x1_pos", np.maximum(x[:,0], 0.0))
    n = iv("x1_neg", np.maximum(-x[:,0], 0.0))
    h = p[:,None]*np.maximum(Wx,0.0) + n[:,None]*np.maximum(-Wx,0.0)
    h = iv("s2", np.maximum(x[:,1:2]*Wx + h @ Wh, 0.0))
    h = iv("s3", np.maximum(x[:,2:3]*Wx + h @ Wh, 0.0))
    h = iv("s4", np.maximum(x[:,3:4]*Wx + h @ Wh, 0.0))
    h = iv("s5", np.maximum(x[:,4:5]*Wx + h @ Wh, 0.0))
    o = h @ Wc
    return np.concatenate([o, -o.sum(1, keepdims=True)], axis=1)

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
