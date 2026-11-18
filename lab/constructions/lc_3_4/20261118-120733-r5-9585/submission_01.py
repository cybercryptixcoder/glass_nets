import numpy as np
DESCRIPTION = ("A 2-layer, one-head, attention-only transformer read as a circuit. The residual stream is 3 wide and "
 "carries token+position embeddings. Layer 0: every position scores every other position bilinearly "
 "(score_ij = h_i M0 h_j^T), softmaxes the scores over keys into its attention row, and adds the "
 "attention-weighted value (VO0) to its own residual. Layer 1 repeats with M1/VO1. The last position's residual "
 "is then read out linearly; class 1 is the reference class and its column is dropped. All numbers are measured "
 "constants of the network; the file never runs the network.")
Em = np.array([[0.20489, 0.45338, 0.15288],
 [2.82839, 2.24345, -0.02893],
 [2.11399, 2.35758, 0.32271],
 [3.10221, 1.58576, 1.26419]])
Pm = np.array([[-0.93149, -2.11544, -0.54955],
 [-1.86509, -1.547, 3.0974],
 [-5.9734, -0.12308, 1.30843]])
M0 = np.array([[0.18668, -0.39445, 0.09827],
 [-0.44822, -1.26299, 1.4457],
 [-2.82766, -1.65353, 2.64965]])
VO0 = np.array([[0.03035, -0.77354, 0.13001],
 [1.06435, -0.10224, 0.24568],
 [0.74572, 1.11353, -0.10662]])
M1 = np.array([[0.25816, 2.9383, 0.26575],
 [-0.19537, -2.23514, -0.21174],
 [-0.27792, 1.27298, 3.80974]])
VO1 = np.array([[-1.87926, 1.5715, -1.81175],
 [0.04509, 0.44171, -0.40485],
 [-0.16925, -9.84681, 5.84598]])
WU = np.array([[37.4455, 0, 35.6038, 36.8856],
 [6.30266, 0, 3.63439, 4.53125],
 [1.06395, 0, -4.666, -2.58602]])

def model(x, iv):
    B = x.shape[0]
    pos = np.vstack([Pm, np.zeros((1,3))])
    h = iv('embed', (Em[x] + pos).reshape(B,-1)).reshape(B,4,3)
    for l,(m,vo) in enumerate(((M0,VO0),(M1,VO1))):
        s = np.einsum('bid,bjd->bij', h@m, h)
        a = np.exp(s - s.max(-1, keepdims=True)); a = a/a.sum(-1, keepdims=True)
        a = iv('pattern%d'%l, a.reshape(B,-1)).reshape(B,4,4)
        o = iv('attn_out%d'%l, np.einsum('bij,bjd->bid', a, h@vo).reshape(B,-1)).reshape(B,4,3)
        h = iv('resid_mid%d'%l, (h+o).reshape(B,-1)).reshape(B,4,3)
    return h[:,3]@WU

ALIGN = {'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],
 'pattern1':['attn1.pattern'],'attn_out1':['attn1.out'],'resid_mid1':['resid1.mid']}
