
import numpy as np
DESCRIPTION = "Attention-only 2-layer net in a token-basis gauge: tokens 0,1,2 are the three residual basis vectors and token 3 is the origin. Position vectors P are added, each layer computes softmax(h M h^T) and adds (attn @ h) V, the last position is read out linearly by U."
E = np.zeros((4,3)); E[:3] = np.eye(3)
P = np.array([[-0.217,0.964,-1.827],[-0.884,-4.495,2.584],[-0.559,-6.036,6.218],[-1.222,-0.313,0.529]])
M0 = np.array([[-8.5940,8.3420,4.9130],[-8.4140,3.2600,0.1770],[-5.6370,1.9640,0.0330]])
V0 = np.array([[-0.0370,-1.4990,2.6720],[0.4780,0.2560,-1.1960],[0.0970,0.0810,-0.3970]])
M1 = np.array([[14.3020,1.3550,-1.0090],[9.6860,4.0220,1.6530],[11.4760,1.3960,-0.5770]])
V1 = np.array([[-2.8380,1.3830,2.2890],[-3.6200,2.0220,9.2820],[-2.6390,1.5770,5.2250]])
U = np.array([[-116.8100,0.0,-102.0850,-109.1260],[-7.4840,0.0,-1.3250,-3.7760],[-33.1410,0.0,-27.9860,-30.5180]])
def model(x, iv):
    B = x.shape[0]
    h = iv('embed', (E[x] + P).reshape(B,-1)).reshape(B,4,3)
    for l,(M,VO) in enumerate([(M0,V0),(M1,V1)]):
        s = h @ M @ h.transpose(0,2,1); s = s - s.max(-1, keepdims=True)
        a = np.exp(s); a = a / a.sum(-1, keepdims=True)
        a = iv('pattern%d'%l, a.reshape(B,-1)).reshape(B,4,4)
        o = iv('attn_out%d'%l, np.einsum('bij,bjk,kl->bil',a,h,VO).reshape(B,-1)).reshape(B,4,3)
        h = iv('resid_mid%d'%l, (h+o).reshape(B,-1)).reshape(B,4,3)
    return h[:,3] @ U
ALIGN = {'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],'pattern1':['attn1.pattern'],'attn_out1':['attn1.out'],'resid_mid1':['resid1.mid']}
