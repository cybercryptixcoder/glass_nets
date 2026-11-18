import numpy as np
DESCRIPTION="""Sequential ReLU integrator; the decision is the position of the value the network believes is second-largest.
RULE 1: h1=relu(x1*W+)+relu(-x1*W-) with W+=relu(Wx), W-=relu(-Wx): the first state is confined to the exact 2-D cone spanned by the positive and negative parts of the input-gain vector, and each unit's input gain is a single number shared by all five steps.
RULE 2: only differences of the five class scores matter, so the readout is centred per hidden unit (its five class weights sum to zero) and class 5 is minus the sum of classes 1-4: 32 numbers instead of 40.
CORE: h_t = relu(x_t*Wx + h_(t-1) @ Wh), a dense 8x8 mixing. Nothing here is removable: dropping any unit or single weight costs >40% of the decisions and rank truncations of Wh or of the readout also fail, so Wh is carried as measured numbers (quantised to 1e-2, readout 1e-2, input gains 1e-2; 97% of decisions kept).
out = h5 @ Wc is a linear read of the final state. Behaviour: agrees with the true second-largest position on 88% of inputs, erring towards the largest or the third when neighbouring order statistics are close."""
Wx=np.array([64,-642,42,-657,5,-491,-5,700])/100.
Wh=np.array([[13,15,48,6,77,4,94,-16],[-46,-24,27,49,17,-73,42,-3],[2,-13,-11,-19,-62,-42,24,8],[53,-30,-36,-85,28,57,20,6],[-18,-32,22,-25,-75,-19,-121,38],[88,60,13,35,13,7,-75,2],[13,96,-9,97,55,68,79,-89],[-166,-2,142,109,16,81,110,-83]])/100.
Wc=np.array([[91,-94,87,-176],[46,-59,-378,401],[113,21,-311,85],[52,-227,496,-270],[-49,207,-56,-56],[-84,424,-132,-127],[-84,-91,54,70],[43,51,57,55]])/100.
def model(x,iv):
    s=np.zeros((len(x),8))
    for t in range(5): s=iv('h%d'%(t+1),np.maximum(x[:,t:t+1]*Wx+s@Wh,0))
    o=s@Wc
    return np.hstack([o,-o.sum(1,keepdims=True)])
ALIGN={'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
