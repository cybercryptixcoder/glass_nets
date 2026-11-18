
import numpy as np, math
DESCRIPTION = ("Mechanism, in effective terms (18 raw weight matrices collapsed to the maps actually used): "
 "residual[p] = token_embedding[t_p] + position_embedding[p]. Layer 0: every position forms a query/key "
 "with the single bilinear score G0 = W_Q W_K^T (only the product matters, so 16 numbers not 32), attends "
 "(temperature 1/2, no mask), and each position adds the attention-weighted average of the other positions' "
 "embeddings passed through M0 = W_V W_O. Layer 1 repeats with G1 and M1 on the new residual; only the "
 "last position's row matters. Readout: last residual dotted with the class-score differences "
 "W_U - W_U[:,3], i.e. five class scores up to a common shift. So: two bilinear attention rounds with "
 "value maps, a token+position embedding, and a linear readout at the last position.")
WE=np.array([1.02219,-0.0738332,0.678463,1.42376,1.89628,-0.640477,0.300952,1.39606,1.14405,0.999148,-0.554679,1.76927,1.8459,-1.08737,2.88155,-0.0166118,1.09393,-0.200079,-2.02604,-0.521553]).reshape(5,4)
WP=np.array([-0.473798,-1.57124,1.90768,-1.67348,0.593856,2.25469,0.0259214,-1.15358,0.169568,-0.525078,2.54645,1.95688,-0.0670782,2.60831,-1.29808,-0.452534,-1.59225,2.06075,-1.15129,0.933021]).reshape(5,4)
G0=np.array([-15.0389,-14.5485,0.180469,-8.09595,-1.18449,-3.14992,3.17739,10.9307,-6.25304,-4.21072,-1.10425,0.719257,-8.30914,2.64618,-0.492964,-22.7083]).reshape(4,4)
G1=np.array([-2.1074,-2.93225,-0.599269,-0.205361,-0.632513,-0.999716,0.0629805,0.890373,-1.21046,1.55985,-0.0674067,1.95849,0.23845,1.069,-0.0424175,-0.931828]).reshape(4,4)
M0=np.array([0.765185,-0.0242773,0.131896,0.16006,-0.40986,-1.00846,0.610157,0.385657,0.33974,-0.813033,-0.0237156,-0.603942,0.516506,1.92233,-0.291387,-0.191244]).reshape(4,4)
M1=np.array([1.05769,2.14696,-1.54633,-0.304346,-1.01914,12.8333,-4.60708,0.657116,-0.173268,13.0053,1.29864,-1.79558,-0.62603,1.16206,-4.76391,2.43607]).reshape(4,4)
D=np.array([-0.576917,-1.10709,-0.68308,0,-0.732706,-0.336999,-0.356207,-1.66284,0,-0.593721,0.388262,-0.335783,-0.499346,0,0.141475,0.28566,0.21341,0.922068,0,0.454917]).reshape(4,5)

def sm(s):
    s=s-s.max(-1,keepdims=True); e=np.exp(s); return e/e.sum(-1,keepdims=True)

def model(x, iv):
    em=WE[x]+WP[None,:,:]
    em=np.stack([iv("e%d"%p, em[:,p,:].copy()) for p in range(5)],1)
    A0=sm((em@G0)@np.swapaxes(em,1,2)*0.5)
    r0=em+(A0@(em@M0))
    r0=np.stack([iv("r0_%d"%p, r0[:,p,:].copy()) for p in range(5)],1)
    A1=sm((r0@G1)@np.swapaxes(r0,1,2)*0.5)
    A1[:,4,:]=iv("a1p4", A1[:,4,:].copy())
    a1=(A1@(r0@M1))
    a1[:,4,:]=iv("a1o4", a1[:,4,:].copy())
    r1=r0+a1
    r1=np.stack([iv("r1_%d"%p, r1[:,p,:].copy()) for p in range(5)],1)
    return r1[:,4,:]@D

ALIGN={"e0":["embed"],"e1":["embed"],"e2":["embed"],"e3":["embed"],"e4":["embed"],
 "r0_0":["resid0.mid"],"r0_1":["resid0.mid"],"r0_2":["resid0.mid"],"r0_3":["resid0.mid"],"r0_4":["resid0.mid"],
 "a1p4":["attn1.pattern"],"a1o4":["attn1.out"],
 "r1_0":["resid1.mid"],"r1_1":["resid1.mid"],"r1_2":["resid1.mid"],"r1_3":["resid1.mid"],"r1_4":["resid1.mid"]}
