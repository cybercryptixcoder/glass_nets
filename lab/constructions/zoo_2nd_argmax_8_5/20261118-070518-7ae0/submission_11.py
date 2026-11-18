import numpy as np

DESCRIPTION = (
 "Runner-up detector: the argmax output is the position of the SECOND largest of the five inputs (87.9%; errors sit "
 "where the 2nd and 3rd largest are within ~0.27). Step 1 is two rays only: h1 = relu(x1)*[s]+ + relu(-x1)*[-s]+, a "
 "2-dimensional state. Each unit j of steps 2-5 asks 'is this input above my threshold?': h_j = relu(s_j*x_t + "
 "(h_{t-1} @ G)_j) with s_j the sign of the unit's input weight. The state is written in the scale-free gauge "
 "h'_j = h_j/|WX_j|, which makes the input drive exactly +-1 (the eight input weights become integer signs) and puts "
 "the whole recurrence into G = diag(|WX|) WH diag(1/|WX|). Reading out with WO = diag(|WX|) WO_shifted, where the "
 "per-unit row-shift freedom (adding c_j to all five class weights of unit j adds c_j*h_j to every class score) makes "
 "class 0 identically zero, leaves 32 readout numbers, i.e. 96 constants for 95 degrees of freedom. Mechanism: units "
 "0,2,4,6 have tiny |WX| and huge scaled activations (up to ~1000; on for 73-99% of inputs) and are amplifying latches; "
 "units 1,3,5,7 stay O(1-6) and are level detectors. The latches drive the detectors with large negative coefficients "
 "(G[0,1]=-4.6, G[0,7]=-18, G[5,6]=-68, G[7,6]=+143) so the assessment is winner-take-all: an input above the current "
 "level switches a latch on and it suppresses the other units until a larger one arrives; the readout reads the level "
 "ladder left in the detectors. Measured dead ends (why this part is still the raw recurrence): Frobenius rank-6 of G "
 "keeps 89% of decisions and rank-5 38%; state-weighted low-rank fits of G stall at 11% drive error and 26% decisions; "
 "zeroing any single one of the 64 couplings flips 0.8-58% of decisions (greedy pruning reaches only 85% after 11 "
 "drops); G has no product/low-rank-in-log structure; and h5 is NOT a function of the two largest inputs (relative "
 "change 0.75 when the top-2 values and the argmax position are matched), so the state depends on the full input order.")

SGN = np.array([1, -1, 1, -1, 1, -1, -1, 1])

G = np.array([
[ 0.125403, 0.015506, 0.722577, 0.005929, 10.991920, 0.004699,  11.213180,-0.014990],
[-4.643681,-0.236367, 4.129188, 0.481472, 24.082860,-0.956997,  50.346780,-0.024794],
[ 0.013155,-0.008510,-0.106897,-0.012254,-5.808311,-0.036525,   1.877949, 0.004936],
[ 5.402196,-0.307102,-5.632124,-0.847493, 40.717220, 0.762250,  24.127280, 0.057456],
[-0.012450,-0.002277, 0.022994,-0.001727,-0.746841,-0.001757,  -1.006344, 0.002450],
[ 6.735964, 0.459419, 1.458299, 0.259920, 13.861720, 0.069581, -68.139580, 0.016556],
[ 0.010670, 0.008102,-0.011046, 0.008006,  0.654784, 0.007482,   0.786684,-0.006881],
[-18.042140,-0.018800,23.456770, 1.158553,24.193110, 1.161252, 142.629400,-0.827896]])

WO = np.array([
[ -1.190532, -0.023341,  -1.712173,  0.007333],
[ -6.758611,-27.233590,  22.787280, -3.696860],
[ -0.388637, -1.793530,  -0.119464, -0.091131],
[-18.313170, 29.212920, -21.142470, -6.789302],
[  0.115567, -0.002906,  -0.002928,  0.001177],
[ 24.914920, -2.356280,  -2.157164,  0.130843],
[ -0.003834,  0.074910,   0.083627,  0.073568],
[  0.542813,  0.965995,   0.812952,-17.454250]])


def model(x, iv):
    p = iv("pos1", np.maximum(x[:, 0], 0))
    n = iv("neg1", np.maximum(-x[:, 0], 0))
    h = p[:, None] * np.maximum(SGN, 0) + n[:, None] * np.maximum(-SGN, 0)
    for t in range(2, 6):
        h = iv("h%d" % t, np.maximum(x[:, t-1:t] * SGN + h @ G, 0))
    out = h @ WO
    return np.concatenate([np.zeros((out.shape[0], 1)), out], axis=1)

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "h5": ["h5"]}
