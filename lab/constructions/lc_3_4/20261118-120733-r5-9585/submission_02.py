import numpy as np
DESCRIPTION = 'Two single-head attention layers, width 3, no MLP, no biases. Read in the basis where tokens 1,2,3 are the coordinate axes (E=[[T],[I]]) and position 0 carries no position vector: each layer gives every position a bilinear score against every other, score_ij=h_i M h_j^T, softmaxes it over keys j, and adds the attention-weighted value (VO) to its residual. The last position is read out linearly, class 1 being the reference (its column is zero). Every number is a measured constant.'
T  = np.array([4.4356,-8.4048,0.36765])
P  = np.array([[-8.4143,10.011,1.4696],[-8.5166,10.919,-0.82833],[-5.7314,10.796,-0.44359]])
M0 = np.array([[4.3435,2.5599,3.3403],[1.4567,0.82558,1.75],[-3.5651,-2.2653,-1.308]])
V0 = np.array([[4.0933,-8.6703,1.0367],[2.229,-4.6709,0.65395],[1.7186,-3.46,0.39913]])
M1 = np.array([[2.768,2.2333,-2.287],[1.2953,1.1563,-1.1336],[-0.0064815,1.8277,-1.0394]])
V1 = np.array([[-6.6325,22.251,-7.9197],[-2.7242,9.9463,-4.0387],[7.4044,-17.459,1.0946]])
WU = np.array([[71.221,0,70.701,72.044],[45.564,0,44.04,45.301],[78.706,0,72.026,75.82]])

def model(x, iv):
    B = x.shape[0]
    h = iv('embed', (np.vstack([T, np.eye(3)])[x] + np.vstack([np.zeros((1,3)), P])).reshape(B,-1)).reshape(B,4,3)
    for l,(m,vo) in enumerate(((M0,V0),(M1,V1))):
        s = np.einsum('bid,bjd->bij', h@m, h)
        a = np.exp(s - s.max(-1,keepdims=True)); a = a/a.sum(-1,keepdims=True)
        a = iv('pattern%d'%l, a.reshape(B,-1)).reshape(B,4,4)
        o = iv('attn_out%d'%l, np.einsum('bij,bjd->bid', a, h@vo).reshape(B,-1)).reshape(B,4,3)
        h = iv('resid_mid%d'%l, (h+o).reshape(B,-1)).reshape(B,4,3)
    return h[:,3]@WU

ALIGN = {'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],
 'pattern1':['attn1.pattern'],'attn_out1':['attn1.out'],'resid_mid1':['resid1.mid']}
