
import numpy as np
DESCRIPTION=(
"How this network computes its answer. It reads 5 standard-normal numbers x1..x5 one per step into an 8-unit ReLU state "
"h_t = relu(x_t * a + h_{t-1} @ B), h_0 = 0, and scores 5 classes by a linear readout out = h5 @ C. "
"The decision (argmax) is 'which position holds the second-largest input': the network agrees with that rank rule 88% of the time, "
"and when it errs it jumps to the largest (42% of errors) or third-largest (52%) position on near-ties, i.e. it is a graded second-largest detector, not a rank sorter. "
"The 8 state units divide by input gain |a| into a large-gain detector group (units 1,3,5,7; |a|=4.9..7.0) that is dominated by the incoming x_t at its own step, "
"and a small-gain integrator group (units 0,2,4,6; |a|=0.04..0.64) that is driven almost entirely by the recurrent mix; units 4 and 6 get essentially no direct input (pure echoes). "
"The recurrent matrix B (64 numbers) is dense and full rank (eigenvalues in 4 complex-conjugate pairs, |.|=0.46..1.19, so the state never dies and never blows up); "
"it cannot be replaced by any low-rank, symmetric, circulant or sparse rule without losing the decision, so it is carried as measured constants. "
"The readout C is exactly centred: each unit's five scores sum to zero, so class 5 = -sum of classes 1-4 and only rank-4 (32) constants are needed (identical decisions). "
"All three matrices (a: 8, B: 64, C: 32 = 104 constants, 3 dp) stay as measured numbers; the only exact rule is the centred, rank-4 readout.")
a=np.array([643,-6421,423,-6574,45,-4913,-54,7002])*.001
B=np.array([125,155,476,61,771,36,944,-163,-465,-236,272,493,169,-732,424,-27,20,-129,-107,-190,-618,-424,240,82,528,-300,-363,-847,279,570,199,61,-177,-324,216,-252,-747,-192,-1207,380,881,600,126,348,127,70,-750,24,127,962,-86,973,546,680,787,-890,-1656,-17,1419,1088,156,815,1102,-828]).reshape(8,8)*.001
C=np.array([908,-944,872,-1756,464,-588,-3777,4013,1130,212,-3105,848,518,-2268,4962,-2698,-492,2071,-556,-557,-836,4235,-1315,-1275,-844,-915,541,702,432,510,570,548]).reshape(8,4)*.001
def model(x,iv):
    s=np.zeros((len(x),8))
    for t in range(5):
        s=iv('h%d'%(t+1), np.maximum(x[:,t:t+1]*a + s@B, 0))  # detector units copy x_t; integrators mix h_{t-1}
    p=s@C                                                  # rank-4 centred readout
    return np.column_stack([p, -p.sum(1)])                 # class 5 = -sum of the other four
ALIGN={'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
