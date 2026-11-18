
import numpy as np
DESCRIPTION=("A one-layer ReLU RNN reads x1..x5 and outputs 5 class scores; decision = argmax = 'which position holds the second-largest input' "
"(the network agrees with that rank rule 88% of the time and its mistakes are jumps to the largest/third rank on near-ties). "
"Dynamics: h_t = relu(x_t*a + h_{t-1}@b), h_0=0. Units split into a large-gain input-detector group (a_j large, ~4.9-7) and a small-gain integrator group (a_j small, ~0.04-0.64); "
"a dense 8-unit linear recurrence mixes them. Readout: out = h5@c with each unit's five scores summing to zero, so class5 = -sum of the other four (identical decisions, 32 constants instead of 40). "
"a,b,c are the measured weights (3 dp).")
a=np.array([643,-6421,423,-6574,45,-4913,-54,7002])*.001
b=np.array([125,155,476,61,771,36,944,-163,-465,-236,272,493,169,-732,424,-27,20,-129,-107,-190,-618,-424,240,82,528,-300,-363,-847,279,570,199,61,-177,-324,216,-252,-747,-192,-1207,380,881,600,126,348,127,70,-750,24,127,962,-86,973,546,680,787,-890,-1656,-17,1419,1088,156,815,1102,-828]).reshape(8,8)*.001
c=np.array([908,-944,872,-1756,464,-588,-3777,4013,1130,212,-3105,848,518,-2268,4962,-2698,-492,2071,-556,-557,-836,4235,-1315,-1275,-844,-915,541,702,432,510,570,548]).reshape(8,4)*.001
def model(x,iv):
    s=np.zeros((len(x),8))
    for t in range(5):s=iv('h%d'%(t+1),np.maximum(x[:,t:t+1]*a+s@b,0))
    p=s@c
    return np.column_stack([p,-p.sum(1)])
ALIGN={'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
