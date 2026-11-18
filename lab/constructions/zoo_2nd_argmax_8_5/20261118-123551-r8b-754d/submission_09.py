
import numpy as np
DESCRIPTION="ReLU RNN: h_t=relu(x_t*a+h_{t-1}@b), h0=0; out=h5@c, each unit's five scores sum to zero so class5=-sum(others) (rank-4 readout). a,b,c = measured weights: input gains a and readout c to 2 dp, recurrent matrix b to 3 dp (the recurrence is the sensitive part)."
a=np.array([64,-642,42,-657,5,-491,-5,700])*.01
b=np.array([125,155,476,61,771,36,944,-163,-465,-236,272,493,169,-732,424,-27,20,-129,-107,-190,-618,-424,240,82,528,-300,-363,-847,279,570,199,61,-177,-324,216,-252,-747,-192,-1207,380,881,600,126,348,127,70,-750,24,127,962,-86,973,546,680,787,-890,-1656,-17,1419,1088,156,815,1102,-828]).reshape(8,8)*.001
c=np.array([91,-94,87,-176,46,-59,-378,401,113,21,-311,85,52,-227,496,-270,-49,207,-56,-56,-84,424,-132,-127,-84,-91,54,70,43,51,57,55]).reshape(8,4)*.01
def model(x,iv):
    s=np.zeros((len(x),8))
    for t in range(5):s=iv('h%d'%(t+1),np.maximum(x[:,t:t+1]*a+s@b,0))
    p=s@c
    return np.column_stack([p,-p.sum(1)])
ALIGN={'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
