
import numpy as np
DESCRIPTION=("A one-layer ReLU RNN reads x1..x5 and outputs 5 class scores; decision = argmax = which position holds the second-largest input "
"(the network agrees with that rank rule 88% and its mistakes are jumps to largest/third rank on near ties). "
"Dynamics h_t=relu(x_t*a+h_{t-1}@b), h_0=0; a dense 8-unit recurrence with large-gain detector units and small-gain integrator units. "
"Readout out=h5@c, each unit's five scores sum to zero so class5=-sum of the other four. a,b,c are the measured weights (2 dp) written as integers/100.")
a=np.array([64,-642,42,-657,5,-491,-5,700])*.01
b=np.array([13,15,48,6,77,4,94,-16,-46,-24,27,49,17,-73,42,-3,2,-13,-11,-19,-62,-42,24,8,53,-30,-36,-85,28,57,20,6,-18,-32,22,-25,-75,-19,-121,38,88,60,13,35,13,7,-75,2,13,96,-9,97,55,68,79,-89,-166,-2,142,109,16,81,110,-83]).reshape(8,8)*.01
c=np.array([91,-94,87,-176,92,46,-59,-378,401,-11,113,21,-311,85,91,52,-227,496,-270,-51,-49,207,-56,-56,-47,-84,424,-132,-127,-81,-84,-91]).reshape(8,4)*.01
def model(x,iv):
    s=np.zeros((len(x),8))
    for t in range(5):s=iv('h%d'%(t+1),np.maximum(x[:,t:t+1]*a+s@b,0))
    p=s@c
    return np.column_stack([p,-p.sum(1)])
ALIGN={'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
