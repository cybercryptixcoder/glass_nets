import numpy as np
DESCRIPTION="""Sequential ReLU integrator; the decision is the position of the (believed) second-largest input.
RULE 1: h1=relu(x1*W+)+relu(-x1*W-) with W+=relu(Wx), W-=relu(-Wx), so the first state lives in the exact 2-D cone spanned by the positive and negative parts of the input-gain vector (the input gain of each unit is a single number, shared by all steps).
RULE 2: only differences of the five class scores matter, so the readout is centred per hidden unit (its five class weights sum to zero) and class 5 is minus the sum of classes 1-4; 32 numbers instead of 40.
Core: h_t = relu(x_t*Wx + h_(t-1) @ Wh): a dense 8x8 mixing. Every unit and every entry matters (removing any one costs >40% of decisions; rank truncation of Wh or Wo also fails), so Wh is carried as measured numbers, quantised to 1e-3 (Wx: 1e-3, readout: 1e-2) which keeps 99.5% of decisions.
Scores out = h5 @ Wc are a linear read of the final state; the network's decision matches the true second-largest position on 88% of inputs and errs towards the largest or third when neighbouring order statistics are close, which this recurrence reproduces."""
Wx=np.array([643,-6421,423,-6574,45,-4913,-54,7002])/1000.
Wh=np.array([[125,155,476,61,771,36,944,-163],[-465,-236,272,493,169,-732,424,-27],[20,-129,-107,-190,-618,-424,240,82],[528,-300,-363,-847,279,570,199,61],[-177,-324,216,-252,-747,-192,-1207,380],[881,600,126,348,127,70,-750,24],[127,962,-86,973,546,680,787,-890],[-1656,-17,1419,1088,156,815,1102,-828]])/1000.
Wc=np.array([[91,-94,87,-176],[46,-59,-378,401],[113,21,-311,85],[52,-227,496,-270],[-49,207,-56,-56],[-84,424,-132,-127],[-84,-91,54,70],[43,51,57,55]])/100.
def model(x,iv):
    s=np.zeros((len(x),8))
    for t in range(5):
        s=iv('h%d'%(t+1),np.maximum(x[:,t:t+1]*Wx+s@Wh,0))
    o=s@Wc
    return np.hstack([o,-o.sum(1,keepdims=True)])
ALIGN={'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
