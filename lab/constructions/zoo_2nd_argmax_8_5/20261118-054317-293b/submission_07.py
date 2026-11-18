import numpy as np
DESCRIPTION = ("Task: the network's decision is a soft version of 'which of the five positions holds the runner-up "
 "(second largest)' - it agrees with that rule on ~88% and its errors sit where the gap between the top two values "
 "is small. Mechanism: five steps of a ReLU integrator h_t = relu(x_t*Wx + h_{t-1}*Wh) with an 8-unit state; "
 "h1 = relu(x1*Wx) = p*relu(Wx) + n*relu(-Wx), a two-ray cone spanned by relu(x1) and relu(-x1). Each later state is "
 "the same integrator applied to the next input. Only score DIFFERENCES matter, so the readout is written centred "
 "(each hidden unit's five class scores sum to zero) and the fifth score is minus the sum of the other four - the "
 "canonical form of a class decision. Two dead input weights (units 4 and 6 carry |Wx|<0.06, i.e. no input drive, "
 "pure integrators) and two tiny recurrent entries are dropped.")

Wx = np.array([0.6426,-6.4213,0.4235,-6.5737,0.0,-4.9132,0.0,7.0017])
Wh = np.array([[ 0.1254, 0.1549, 0.4761, 0.0606, 0.7712, 0.0359, 0.9440,-0.1633],
 [-0.4647,-0.2364, 0.2723, 0.4929, 0.1691,-0.7322, 0.4242,-0.0270],
 [ 0.0,   -0.1290,-0.1069,-0.1902,-0.6185,-0.4238, 0.2399, 0.0816],
 [ 0.5281,-0.3000,-0.3628,-0.8475, 0.2793, 0.5697, 0.1986, 0.0612],
 [-0.1774,-0.3243, 0.2159,-0.2517,-0.7468,-0.1915,-1.2074, 0.3804],
 [ 0.8810, 0.6004, 0.1257, 0.3478, 0.1272, 0.0696,-0.7503, 0.0236],
 [ 0.1268, 0.9617,-0.0865, 0.9728, 0.5457, 0.6795, 0.7867,-0.8905],
 [-1.6559, 0.0,    1.4186, 1.0877, 0.1558, 0.8149, 1.1020,-0.8279]])
# centred readout: columns of  Wo - rowmean(Wo), class 4 dropped (recovered as minus the sum)
Wc = np.array([[ 0.90838,-0.94422, 0.87208,-1.75602],
 [ 0.46418,-0.58842,-3.77702, 4.01288],
 [ 1.13008, 0.21238,-3.10532, 0.84798],
 [ 0.51818,-2.26762, 4.96208,-2.69802],
 [-0.49194, 2.07116,-0.55644,-0.55694],
 [-0.83586, 4.23524,-1.31536,-1.27486],
 [-0.84390,-0.91480, 0.54080, 0.70190],
 [ 0.43224, 0.50974, 0.57024, 0.54834]])

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
