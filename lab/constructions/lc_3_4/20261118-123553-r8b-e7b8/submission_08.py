import numpy as np
DESCRIPTION=("h=E[token]+P[position] (residual width 3, P[3]=0). Two attention layers, one head each: "
 "s=softmax_j(h_i.M.h_j^T), h=h+s.h.V. logits=h[pos3]@U (U column 1 = 0 by a gauge). "
 "All 66 constants are the network's own matrices at 2 significant digits, refitted so the decisions agree.")
T=np.array([0.18,0.48,0.16,2.8,2.2,-0.029,2.1,2.4,0.32,3.1,1.6,1.3,-0.98,-2.1,-0.57,-1.9,-1.6,3.1,-6,-0.15,1.3,0.19,-0.37,0.12,-0.43,-1.3,1.4,-2.7,-1.5,2.6,0.024,-0.75,0.14,1.1,-0.067,0.24,0.75,1.1,-0.11,0.26,2.9,0.28,-0.2,-2.2,-0.21,-0.29,1.2,3.8,-1.9,1.5,-1.8,0.039,0.38,-0.43,-0.15,-9.6,6,37,35,34,6.4,3.6,4.3,1.1,-4.8,-2.2]).reshape(22,3)
E,P=T[:4],np.vstack([T[4:7],[0,0,0]])
U=np.zeros((3,4));U[:,[0,2,3]]=T[19:22]
M0,V0,M1,V1=T[7:10],T[10:13],T[13:16],T[16:19]
def model(x,iv):
    h=iv('e',(E[x]+P).reshape(len(x),12)).reshape(-1,4,3)
    for k,(M,V) in enumerate(((M0,V0),(M1,V1))):
        s=h@M@h.transpose(0,2,1)
        s=np.exp(s-s.max(-1,keepdims=True));s/=s.sum(-1,keepdims=True)
        h=iv('r'+'01'[k],(h+s@h@V).reshape(len(x),12)).reshape(-1,4,3)
    return h[:,3]@U
ALIGN={'e':['embed'],'r0':['resid0.mid'],'r1':['resid1.mid']}
