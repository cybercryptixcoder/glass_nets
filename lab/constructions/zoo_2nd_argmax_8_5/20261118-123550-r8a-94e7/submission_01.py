import numpy as np
DESCRIPTION = """Faithful account of the network.
Reading: the recurrence is expressed in the units' own scale: divide hidden unit j (and multiply
its readout row) by |W_x[j]|. In that gauge every input gain has magnitude 1 (signs +,-,+,-,+,-,-,+),
so h1 = relu(x1)*[signs]+ + relu(-x1)*[signs]- lies in an exact 2-D cone (RULE 1: the input gains of
the eight units differ only in sign).
The rest is a dense ReLU integrator: h_t = relu(x_t*(+-1) + h_(t-1) @ WH); no unit, weight or rank
truncation of WH is removable (each removal loses >40% of the decisions), so the 8x8 mixing matrix is
carried as measured numbers.
Readout: only differences of the five class scores matter, so the readout is centred per hidden unit
(its five class weights sum to zero) and the fifth class is recovered as minus the sum (RULE 2).
The scores are a linear read of the final state; the decision is the largest score, i.e. the position
the network believes is second-largest (it agrees with the true second largest on 88% of inputs and
errs towards the largest/third when neighbouring order statistics are close)."""

SGN = np.array([1., -1., 1., -1., 1., -1., -1., 1.])
WH = np.array([[0.12540257, 0.01550624, 0.72257617, 0.00592875, 10.99196107, 0.00469862, 11.21315261, -0.01499044],
 [-4.64366248, -0.23636714, 4.12918173, 0.48147249, 24.08297926, -0.95699546, 50.34678655, -0.02479358],
 [0.01315482, -0.00850972, -0.10689737, -0.01225325, -5.80834771, -0.03652504, 1.87795082, 0.00493648],
 [5.40221872, -0.30711431, -5.63212737, -0.84749269, 40.71743618, 0.76225076, 24.12722671, 0.0574564],
 [-0.01244972, -0.00227694, 0.02299346, -0.0017265, -0.74684125, -0.00175741, -1.006339, 0.00244965],
 [6.73596089, 0.45941631, 1.45830164, 0.25992024, 13.86175685, 0.06958129, -68.13949161, 0.01655681],
 [0.01067084, 0.00810205, -0.01104544, 0.00800533, 0.65478673, 0.00748191, 0.78668416, -0.00688021],
 [-18.042058, -0.0188003, 23.4567915, 1.1585587, 24.19329871, 1.16125149, 142.62923088, -0.82789576]])
WC4 = np.array([[0.58374, -0.606791, 0.560399, -1.12842],
 [2.980358, -3.778255, -24.253268, 25.767637],
 [0.478552, 0.089915, -1.314977, 0.359088],
 [3.406394, -14.906777, 32.619317, -17.736067],
 [-0.022182, 0.093384, -0.025088, -0.02511],
 [-4.106461, 20.808452, -6.462744, -6.263626],
 [-0.045654, -0.049488, 0.029255, 0.037973],
 [3.026497, 3.569312, 3.992491, 3.839451]])

def model(x, iv):
    s = np.zeros((len(x), 8))
    for t in range(5):
        s = iv('h%d' % (t + 1), np.maximum(x[:, t:t + 1] * SGN + s @ WH, 0))
    o4 = s @ WC4
    return np.hstack([o4, -o4.sum(1, keepdims=True)])

ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
