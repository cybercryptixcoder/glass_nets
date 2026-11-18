import numpy as np
DESCRIPTION = """A 5-step ReLU integrator with no biases, out = h5 @ W_o, decision = largest score.
The task it approximates: report the POSITION OF THE SECOND-LARGEST input (88% agreement);
its residual mistakes follow the sign pattern -- when all five inputs are negative it mostly
reports the MOST negative (68%), when exactly two are positive it is most accurate (95%).
RULE 1 (first step): h0 = 0, so h1 = relu(x1*W_x) = relu(x1)*P + relu(-x1)*N with P = relu(W_x),
  N = relu(-W_x). The first state carries exactly TWO numbers: the positive and the negative
  part of x1 (each unit has one input gain, of one sign).
RULE 2 (second step): h2 = relu(x2*W_x + relu(x1)*P2 + relu(-x1)*N2), P2 = P@W_h, N2 = N@W_h:
  the same two numbers pushed through two fixed vectors, plus the new input.
RULE 3 (steps 3-5): the same ReLU recurrence with a dense 8x8 matrix W_h; it cannot be
  compressed (rank truncation, unit pruning and dropping small entries all destroy the
  decisions), so it is written as measured numbers rounded to 1e-3.
RULE 4 (readout): only differences of the five class scores matter, so W_o is centred per
  hidden unit (its five class weights sum to zero) and class 5 is minus the sum of classes
  1-4: 32 numbers instead of 40. Scores are a linear read of the final state.
Fidelity 99.7% of the network's decisions."""
Wx=np.array([643,-6421,423,-6574,45,-4913,-54,7002])/1000.
P2=np.array([-11.513051,-0.090417,10.2033,7.563139,1.290869,5.540416,8.369717,-5.849913])
N2=np.array([4.822943,-0.487783,-0.023582,-0.644849,3.576232,-0.578189,0.385243,0.296448])
Wh=np.array([[125,155,476,61,771,36,944,-163],[-465,-236,272,493,169,-732,424,-27],[20,-129,-107,-190,-618,-424,240,82],[528,-300,-363,-847,279,570,199,61],[-177,-324,216,-252,-747,-192,-1207,380],[881,600,126,348,127,70,-750,24],[127,962,-86,973,546,680,787,-890],[-1656,-17,1419,1088,156,815,1102,-828]])/1000.
Woc=np.array([[908,-944,872,-1756,920],[464,-588,-3777,4013,-112],[1130,212,-3105,848,915],[518,-2268,4962,-2698,-515],[-492,2071,-556,-557,-466],[-836,4235,-1315,-1275,-809],[-844,-915,541,702,516],[432,510,570,548,-2061]])/1000.
def model(x, iv):
    q1 = iv('x1pos', np.maximum(x[:,0],0))
    q2 = iv('x1neg', np.maximum(-x[:,0],0))
    s = iv('h2', np.maximum(x[:,1:2]*Wx + q1[:,None]*P2 + q2[:,None]*N2, 0))
    s = iv('h3', np.maximum(x[:,2:3]*Wx + s@Wh, 0))
    s = iv('h4', np.maximum(x[:,3:4]*Wx + s@Wh, 0))
    s = iv('h5', np.maximum(x[:,4:5]*Wx + s@Wh, 0))
    o = s@Woc
    return np.hstack([o, -o.sum(1, keepdims=True)])
ALIGN={'x1pos':['h1'],'x1neg':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
