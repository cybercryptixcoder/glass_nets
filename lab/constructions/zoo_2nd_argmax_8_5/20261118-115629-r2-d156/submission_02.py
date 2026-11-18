import numpy as np
DESCRIPTION = ("One-layer ReLU recurrent net computing, approximately, WHICH POSITION HOLDS THE SECOND-LARGEST INPUT "
 "(agrees with the task on 88%). Three rules make the account shorter than the weights: (i) the 8 input weights "
 "have arbitrary magnitude - their size is a pure per-unit gauge of the state, so only their SIGNS matter and they "
 "are set to +-1 (this costs no numbers); in these units four of the eight units are input-dominated and four are "
 "recurrent-dominated memory units, and units 4 and 6 are almost never clipped, i.e. they are near-linear "
 "integrators; (ii) only DIFFERENCES of the five class scores decide, so the readout may be made zero-sum and the "
 "5th class is minus the sum of the other four (a rank-1 gauge of the readout, free); (iii) the recurrent matrix "
 "has four complex-conjugate eigenvalue pairs, two of them close to the unit circle, so the state rotates with each "
 "step and its phase tags the position of the input. What remains is a dense 8x8 recurrent matrix A, stored to "
 "3 significant digits, plus the 8x4 zero-sum readout, to 4.")
sg = np.array([1,-1]*3+[-1,1],float)
A = np.array([0.125,0.0155,0.723,0.00593,11,0.0047,11.2,-0.015,-4.64,-0.236,4.13,0.481,24.1,-0.957,50.3,-0.0248,0.0132,-0.00851,-0.107,-0.0123,-5.81,-0.0365,1.88,0.00494,5.4,-0.307,-5.63,-0.847,40.7,0.762,24.1,0.0575,-0.0124,-0.00228,0.023,-0.00173,-0.747,-0.00176,-1.01,0.00245,6.74,0.459,1.46,0.26,13.9,0.0696,-68.1,0.0166,0.0107,0.0081,-0.011,0.00801,0.655,0.00748,0.787,-0.00688,-18,-0.0188,23.5,1.16,24.2,1.16,143,-0.828]).reshape(8,8)
Wc = np.array([0.5837,-0.6068,0.5604,-1.128,2.98,-3.778,-24.25,25.77,0.4786,0.08992,-1.315,0.3591,3.406,-14.91,32.62,-17.74,-0.02218,0.09338,-0.02509,-0.02511,-4.106,20.81,-6.463,-6.264,-0.04565,-0.04949,0.02925,0.03797,3.026,3.569,3.992,3.839]).reshape(8,4)
def model(x, iv):
    s = np.zeros((len(x), 8))
    for t in range(5):
        s = iv('h%d' % (t+1), np.maximum(x[:, t:t+1]*sg + s @ A, 0))
    o = s @ Wc
    return np.c_[o, -o.sum(1)]
ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
