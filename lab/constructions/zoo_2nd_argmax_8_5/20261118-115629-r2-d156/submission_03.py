import numpy as np
DESCRIPTION = ("ReLU RNN whose decision approximates 'which of the 5 inputs is the 2nd largest' (that rule alone gets 88%). "
"Rules: (1) the 8 input weights have arbitrary magnitude - their size is a pure per-unit rescale of the state, so only "
"their signs matter and they are +-1 (free); in these units 4 units are input-dominated, 4 are recurrent memory units, "
"and units 4,6 are never clipped (near-linear); the recurrent matrix has four complex-conjugate eigenvalue pairs, two near "
"the unit circle, so the state rotates at each step and its phase tags the position. (2) only class-score differences "
"decide, so the readout is zero-sum and the 5th score is minus the sum of the other four (free). Left: 8x8 recurrence "
"(5 s.d.) and 8x4 readout (6 s.d.), copied.")
A = np.array([0.1254,0.015506,0.72258,0.0059287,10.992,0.0046986,11.213,-0.01499,-4.6437,-0.23637,4.1292,0.48147,24.083,-0.957,50.347,-0.024794,0.013155,-0.0085097,-0.1069,-0.012253,-5.8083,-0.036525,1.878,0.0049365,5.4022,-0.30711,-5.6321,-0.84749,40.717,0.76225,24.127,0.057456,-0.01245,-0.0022769,0.022993,-0.0017265,-0.74684,-0.0017574,-1.0063,0.0024496,6.736,0.45942,1.4583,0.25992,13.862,0.069581,-68.139,0.016557,0.010671,0.0081021,-0.011045,0.0080053,0.65479,0.0074819,0.78668,-0.0068802,-18.042,-0.0188,23.457,1.1586,24.193,1.1613,142.63,-0.8279]).reshape(8,8)
W = np.array([0.58374,-0.606791,0.560399,-1.12842,2.98036,-3.77826,-24.2533,25.7676,0.478552,0.0899151,-1.31498,0.359088,3.40639,-14.9068,32.6193,-17.7361,-0.0221816,0.0933843,-0.0250878,-0.0251099,-4.10646,20.8085,-6.46274,-6.26363,-0.0456542,-0.0494879,0.0292548,0.0379729,3.0265,3.56931,3.99249,3.83945]).reshape(8,4)
sg = np.array([1,-1]*3+[-1,1],float)
def model(x, iv):
    h = np.zeros((len(x), 8))
    for t in range(5):
        h = iv('h%d' % (t+1), np.maximum(x[:, t:t+1]*sg + h @ A, 0))
    o = h @ W
    return np.c_[o, -o.sum(1)]
ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
