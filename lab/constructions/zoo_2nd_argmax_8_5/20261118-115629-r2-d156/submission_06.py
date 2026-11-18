import numpy as np
DESCRIPTION = ("A one-layer ReLU RNN, 8 units, 5 steps: h_t = relu(x_t*w_x + h_{t-1} W_h) [B,8], out = h_5 W_o [B,5], "
"decision = argmax. The answer is approximately WHICH POSITION (class 1..5) HOLDS THE SECOND-LARGEST INPUT: that rule alone "
"scores 88%, and the network's own 11% of errors are near-ties where the 1st/2nd/3rd largest are almost equal, so it is a "
"soft comparison, not a lookup. HOW IT COMPUTES IT. (a) Step 1 is a two-ray cone: h_1 = relu(x_1)*u + relu(-x_1)*v, so after "
"one input the state knows only 'how positive and how negative x_1 was', in two directions. (b) Units come in two families: "
"four that read x with weight 1 (the odd-indexed ones in this ordering, which get -x) and are input-dominated, and four "
"memory units driven almost entirely by the state; units 4 and 6 are active on 99%/98% of steps, i.e. they are near-linear "
"integrators that never clip. (c) The 8x8 recurrent matrix has four complex-conjugate eigenvalue pairs, two of them close to "
"the unit circle (|lambda| ~ 1.0-1.2, periods 2.5 and 4.3 steps), so the state ROTATES once per step: the phase of the state "
"is a clock that tags which time-step (= position) an input entered at, and the amplitude records its value. The comparison "
"'is this value larger than everything before' is then done by how the rotating amplitude adds to the previous state, and the "
"readout picks the class whose phase matches 'second largest'. WHAT IS RULE AND WHAT IS STILL COPIED. Rule 1 (free): the 8 "
"magnitudes of w_x are a pure per-unit scale of the state - writing g_j = h_j/|w_x[j]| makes w_x the sign vector and rescales "
"W_h and W_o by the same diagonal, with identical outputs - so only the signs matter, and after reordering units they simply "
"alternate (+,-,+,-,...): these 8 constants cost nothing. Rule 2 (free): only differences of the 5 scores decide, so each row "
"of W_o may be shifted by a constant; choosing the shift so rows sum to zero leaves 8x4 numbers and the 5th score is minus the "
"sum of the other four: 8 constants free. Rules 3: units 4 and 6 need no ReLU. What remains - the dense 8x8 recurrence A and "
"the 8x4 readout - is still copied, each entry at the precision it needs (1-7 s.d.). No further compression was found: every "
"one of the 64 recurrent entries is needed (zeroing any single one breaks exactness), every unit is essential (zeroing one "
"loses ~60% of decisions), and rank-7 truncation of the recurrence already loses 10% of decisions - the network sits on a "
"near-critical, oscillatory dynamic where fine details set the near-tie boundary.")
sg = 1 - 2*(np.arange(8) % 2)
A = np.array([0.125,0.0155,0.723,0.00593,10.99,0.0047,-0.015,11.213,-4.64,-0.2364,4.129,0.4815,24.1,-0.957,-0.025,50.35,0.013,-0.0085,-0.107,-0.01225,-5.81,-0.0365,0.0049,1.88,5.4,-0.3071,-5.63,-0.8475,40.72,0.7623,0.0575,24.13,-0.01245,-0.002277,0.023,-0.001727,-0.747,-0.001757,0.00245,-1.006,6.736,0.4594,1.46,0.26,13.86,0.07,0.017,-68.14,-18,-0.02,23.46,1.1586,24.2,1.161,-0.828,142.6,0.0107,0.0081,-0.01105,0.008005,0.655,0.00748,-0.00688,0.7867]).reshape(8,8)
W = np.array([0.584,-0.607,0.5604,-1.128,2.98,-3.78,-24.25,25.77,0.4786,0.09,-1.315,0.359,3.406,-14.9,32.62,-17.74,-0.0222,0.0934,-0.0251,-0.0251,-4.11,20.81,-6.46,-6.26,3.03,3.57,3.99,3.84,-0.04565,-0.0495,0.0292,0.038]).reshape(8,4)
def model(x, iv):
    h = np.zeros((len(x), 8))
    for t in range(5):
        h = iv('h%d' % (t+1), np.maximum(x[:, t:t+1]*sg + h @ A, 0))
    o = h @ W
    return np.c_[o, -o.sum(1)]
ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
