import numpy as np
DESCRIPTION = ("A one-layer ReLU RNN, 8 units, 5 steps: h_t = relu(x_t*w_x + h_{t-1} W_h) in [B,8], out = h_5 W_o in [B,5], decision = argmax "
"(which position, 1-5, holds the 2nd largest input: that rule alone scores 88%, and the net's own 11% of errors are near-ties "
"between the 2nd/3rd/1st largest). Three facts reduce the 112 weights to 96 numbers: "
"RULE 1 (free rescaling). The 8 magnitudes of w_x are a pure per-unit scale of the state: writing g_j = h_j/|w_x[j]| turns "
"w_x into the sign vector sg (+-1 here, 4 negative, 4 positive) and rescales W_h by diag(|w_x|) and W_o by diag(|w_x|) - "
"identical outputs, so the magnitudes carry no information. In these units 4 units read x with weight 1 and are "
"input-dominated, 4 are recurrent memory units, and units 4 and 6 are active 99%/98% of steps, i.e. near-linear integrators. "
"RULE 2 (free readout offset). Only differences of the 5 class scores decide, so each row of W_o may have any constant added; "
"choosing it so that rows sum to zero leaves 8x4 numbers and gives the 5th score as minus the sum of the other four. "
"RULE 3 (dynamics). The recurrent matrix has four complex-conjugate eigenvalue pairs, two of them close to the unit circle, so "
"the state rotates each step and its phase tags the position of the input; that is how a position-blind recurrence can answer "
"a question about positions. What is left - the dense 8x8 recurrence A (no entry could be deleted: all 64 are needed for "
"exactness) and the 8x4 readout - is still copied, each entry to the precision it needs (1-7 s.d.); no low-rank, sparse or "
"symmetric structure of A survives (rank-7 truncation already loses 10% of decisions, zeroing any unit loses 60%).")
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
