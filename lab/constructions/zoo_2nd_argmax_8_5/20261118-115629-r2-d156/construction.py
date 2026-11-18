import numpy as np
DESCRIPTION = ("ReLU RNN, 8 units, 5 steps. Its decision approximates WHICH OF THE 5 INPUTS IS THE SECOND LARGEST "
"(that rule alone scores 88%; the net's own errors are near-ties between 2nd and 3rd/1st). Rules behind the numbers: "
"(1) the 8 input weights have arbitrary magnitudes; their sizes are a pure per-unit rescale of the state, so only their "
"SIGNS matter, and sg is +-1 (free). In these units 4 units are input-driven (large |w|) and read x negatively, 4 are "
"recurrent memory units; units 4,6 are active 99% of steps, i.e. near-linear integrators; the recurrent matrix has four "
"complex-conjugate eigenvalue pairs, two near the unit circle, so the state rotates once per step and its phase tags "
"position. (2) only differences of the 5 class scores decide, so the readout is zero-sum: the 5th score = minus the sum "
"of the other four (free gauge). What is left is the dense 8x8 recurrence A (cycle-free: no entry can be dropped) and "
"the effective 8x4 readout, stored to the precision each entry needs (2-7 s.d.).")
sg = np.array([1,-1]*3+[-1,1],float)
A = np.array([0.1254,0.0155,0.723,0.0059,10.99,0.0047,11.213,-0.015,-4.64,-0.2364,4.129,0.4815,24.1,-0.957,50.35,-0.025,0.013,-0.00851,-0.107,-0.01225,-5.81,-0.03653,1.88,0.005,5.4,-0.3071,-5.63,-0.8475,40.7,0.7623,24.13,0.0575,-0.01245,-0.002277,0.023,-0.001727,-0.7468,-0.001757,-1.0063,0.00245,6.736,0.4594,1.46,0.26,13.86,0.07,-68.14,0.0166,0.01067,0.008102,-0.01105,0.008005,0.655,0.00748,0.7867,-0.00688,-18,-0.02,23.46,1.1586,24.2,1.161,142.6,-0.83]).reshape(8,8)
W = np.array([0.5837,-0.6068,0.5604,-1.128,2.98,-3.778,-24.25,25.768,0.4786,0.09,-1.315,0.359,3.406,-14.91,32.62,-17.736,-0.0222,0.0934,-0.0251,-0.0251,-4.11,20.81,-6.46,-6.264,-0.04565,-0.0495,0.02925,0.038,3.03,3.57,4,3.84]).reshape(8,4)
def model(x, iv):
    h = np.zeros((len(x), 8))
    for t in range(5):
        h = iv('h%d' % (t+1), np.maximum(x[:, t:t+1]*sg + h @ A, 0))
    o = h @ W
    return np.c_[o, -o.sum(1)]
ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
