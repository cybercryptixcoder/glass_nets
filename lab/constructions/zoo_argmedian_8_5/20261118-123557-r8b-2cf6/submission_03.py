import numpy as np
DESCRIPTION = "Five-step ReLU state machine, weights in thousandths, no biases (only the direction of x matters). h1 = relu(x1)*max(W_x,0)+relu(-x1)*max(-W_x,0) is rank 2: the sign of x1 picks one of two rays. Then h_t = relu(x_t*W_x + h_(t-1)@W_h): the same eight input gains, mixed by W_h, rectified. Units 0,5,6 never switch off (linear), units 1,2,4,7 switch half the time. Decision = argmax of h5@W_o."
P = np.array([-174,1514,1499,232,2433,-28,217,-1158,1648,-1680,-1677,491,-3000,421,102,1721,1485,-76,-660,1809,56,530,964,266,-1100,-117,7,-746,-129,-1672,-397,-178,-1120,455,353,-555,808,379,-259,-439,2015,-384,143,-673,-393,984,2191,527,1842,540,465,-366,997,-251,-370,-742,-945,1074,1243,436,1998,-87,179,-1113,614,1694,942,38,1562,86,1992,-483,-424,137,184,117,33,1027,1188,1531,-9184,2049,-1196,-1586,3618,589,-1984,-176,107,-2219,1302,-264,276,426,-2655,5595,-4826,135,-2056,228,667,507,364,208,-126,-2422,984,1373,1326,1189,930,-5514])/1000.
W_x = P[:8][None]; W_h = P[8:72].reshape(8,8); W_o = P[72:].reshape(8,5)

def model(x, iv):
    a = iv("x1pos", np.maximum(x[:,0],0.))
    b = iv("x1neg", np.maximum(-x[:,0],0.))
    s = iv("h1", a[:,None]*np.maximum(W_x,0.) + b[:,None]*np.maximum(-W_x,0.))
    for t in range(1,5):
        s = iv("h%d"%(t+1), np.maximum(x[:,t:t+1]*W_x + s@W_h, 0.))
    return s@W_o

ALIGN = {"x1pos":["h1"],"x1neg":["h1"],"h2":["h2"],"h3":["h3"],"h4":["h4"],"h5":["h5"]}