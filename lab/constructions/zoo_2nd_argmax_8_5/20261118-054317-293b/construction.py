import numpy as np
DESCRIPTION = ("Task: the decision is a soft version of 'which of the five positions holds the runner-up (second largest)'; "
 "the network agrees with that rule on nearly nine tenths of inputs and errs where the gap between the top two values is "
 "small. Mechanism: five steps of a ReLU integrator h_t = relu(x_t*Wx + h_(t-1)*Wh) over an eight-unit state. Stage one is "
 "a two-ray cone, h1 = relu(x1)*relu(Wx) + relu(-x1)*relu(-Wx): the first step only sets the sign pattern of the state. "
 "Every unit is rescaled so that its input weight is exactly plus or minus one (a positive unit rescaling is a symmetry of "
 "relu, since both the input drive and the recurrent rows scale together), which puts all input weights in the integers and "
 "leaves the recurrent matrix and the readout as the only numbers. Units one, three, five and seven are driven hard by the "
 "input and act as switches; units zero, two, four and six are pure recurrent integrators. Only score DIFFERENCES matter, so "
 "the readout is centred (each unit's five class scores sum to zero) and the fifth score is minus the sum of the first four. "
 "Numbers that the decision cannot distinguish, because their slots enter with the same sign and nearly the same magnitude, "
 "are shared (named s1..s11); the file therefore carries fewer distinct numbers than the network has weights.")

s1,s2,s3 = -0.025, 24.134, 3.916
s4,s5,s6 = 0.0048, 3.003, 1.160
s7,s8,s9 = -0.0124, 0.480, 0.0149
s10,s11 = -0.00175, -6.363
Wx = np.array([1,-1,1,-1,1,-1,-1,1])   # sign of each unit's input weight; magnitudes are in the unit scaling
Wh = np.array([[ 0.1254, 0.0155, 0.7226, 0.0059, 10.9920, s4,     11.2132,-0.0150],
 [-4.6437,-0.2364, 4.1292, s8,      s2,     -0.9570, 50.3468, s1     ],
 [ s9,    -0.0085,-0.1069, s7,     -5.8083, -0.0365,  1.8780,  s4    ],
 [ 5.4022,-0.3071,-5.6321,-0.8475, 40.7174,  0.7623,  s2,      0.0575],
 [ s7,    -0.0023, 0.0230, s10,    -0.7468,  s10,    -1.0063,  0.0024],
 [ 6.7360, 0.4594, 1.4583, 0.2599, 13.8618,  0.0696,-68.1395,  s9    ],
 [ 0.0107, 0.0081,-0.0110, 0.0080,  0.6548,  0.0075,  0.7867, -0.0069],
 [-18.0421,-0.0188,23.4568, s6,     s2,      s6,    142.6292,-0.8279]])
Wc = np.array([[ 0.5837,-0.6068,  0.5604, -1.1284],
 [ s5,    -3.7783,-24.2533, 25.7676],
 [ s8,     0.0899, -1.3150,  0.3591],
 [ 3.4064,-14.9068, 32.6193,-17.7361],
 [-0.0222, 0.0934,  s1,       s1    ],
 [-4.1065, 20.8085, s11,      s11   ],
 [-0.0457,-0.0495,  0.0293,  0.0380],
 [ s5,     3.5693,  s3,       s3    ]])

def model(x, iv):
    p = iv("x1_pos", np.maximum(x[:,0], 0))
    n = iv("x1_neg", np.maximum(-x[:,0], 0))
    h = p[:,None]*np.maximum(Wx,0) + n[:,None]*np.maximum(-Wx,0)
    h = iv("s2", np.maximum(x[:,1:2]*Wx + h @ Wh, 0))
    h = iv("s3", np.maximum(x[:,2:3]*Wx + h @ Wh, 0))
    h = iv("s4", np.maximum(x[:,3:4]*Wx + h @ Wh, 0))
    h = iv("s5", np.maximum(x[:,4:5]*Wx + h @ Wh, 0))
    o = h @ Wc
    return np.concatenate([o, -o.sum(1, keepdims=True)], axis=1)

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"], "s5": ["h5"]}
